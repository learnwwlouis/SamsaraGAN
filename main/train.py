import os
import time
import logging
from glob import glob
from datetime import datetime
import scipy.misc
import tensorflow as tf
import numpy as np
from utils import ImagePool
from model import SamsaraGAN
from generator import Generator
from test_model import test_model
from load_train_data import load_train_data

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 256, 'image size, default: 256')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_integer('lambda1', 10,
                        'weight for forward cycle loss (X->Y->X), default: 10.0')
tf.flags.DEFINE_integer('lambda2', 10,
                        'weight for backward cycle loss (Y->X->Y), default: 10.0')
tf.flags.DEFINE_float('learning_rate', 0.0002,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 32,
                        'number of gen filters in first conv layer, default: 32')

tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. horse2zebra_batchsize=1_LR=0.0002_20180206-1441), default: None')

tf.flags.DEFINE_string('dataset_dir', 'apple2orange', 
                       'path of the dataset, default: horse2zebra')

tf.flags.DEFINE_integer('class_num', 2, 
                       'number of classes, default: 2')

tf.flags.DEFINE_integer('iteration', 200000, 
                       'number of iteration, default: 200000')

tf.flags.DEFINE_integer('initial_step', 500, 
                       'number of initial step, default: 500')

def train():
  # Use in discriminator and loss & Append in generator's feature map
  x_class = np.zeros((FLAGS.batch_size, FLAGS.class_num), dtype=np.float32)
  y_class = np.zeros((FLAGS.batch_size, FLAGS.class_num), dtype=np.float32)
  for index in range(FLAGS.batch_size):
    x_class[index][0] = 1.0
    x_class[index][1] = 0.0
  for index in range(FLAGS.batch_size):
    y_class[index][0] = 0.0
    y_class[index][1] = 1.0

  if FLAGS.load_model is not None:
    checkpoints_dir = "checkpoints/" + FLAGS.load_model
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = "checkpoints/{}_batchsize={}_LR={}_{}".format(\
                      FLAGS.dataset_dir, FLAGS.batch_size, FLAGS.learning_rate, current_time)
    try:
      os.makedirs(checkpoints_dir)
    except os.error:
      pass

  graph = tf.Graph()
  with graph.as_default():
    samsara_gan = SamsaraGAN(
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.image_size,
        norm=FLAGS.norm,
        lambda1=FLAGS.lambda1,
        lambda2=FLAGS.lambda1,
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        ngf=FLAGS.ngf,
        class_num=FLAGS.class_num
    )
    G_loss, D_Y_loss, D_X_loss, fake_y, fake_x, Samsara_summary = samsara_gan.model()

    # Summary All trainable variables
    model_vars = tf.trainable_variables()
    tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    optimizers = samsara_gan.optimize(G_loss, D_Y_loss, D_X_loss)
    initialize_G_model, initialize_G_model_summary = samsara_gan.initialize_G_model()

    summary_ini = tf.summary.merge([initialize_G_model_summary])
    summary_op  = tf.summary.merge([Samsara_summary])

    train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    saver = tf.train.Saver(max_to_keep=None)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(graph=graph, config=config) as sess:
    if FLAGS.load_model is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      step = int(meta_graph_path.split("-")[2].split(".")[0])
    else:
      sess.run(tf.global_variables_initializer())
      step = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      fake_Y_pool = ImagePool(FLAGS.pool_size)
      fake_X_pool = ImagePool(FLAGS.pool_size)
      dataA = glob('../data/{}/*.jpg'.format(FLAGS.dataset_dir+'/trainA'))
      dataB = glob('../data/{}/*.jpg'.format(FLAGS.dataset_dir+'/trainB'))
      num_A = len(dataA)
      num_B = len(dataB)
      iter_per_epoch = int( min(num_A, num_B) // FLAGS.batch_size )

      while not coord.should_stop() and step < FLAGS.iteration:
        idx = step % iter_per_epoch
        if idx == 0:
          dataA = glob('../data/{}/*.jpg'.format(FLAGS.dataset_dir+'/trainA'))
          dataB = glob('../data/{}/*.jpg'.format(FLAGS.dataset_dir+'/trainB'))
          np.random.shuffle(dataA)
          np.random.shuffle(dataB)

        batch_image = load_train_data(dataA, dataB, idx, FLAGS.image_size, FLAGS.batch_size)

        # ===================== Initialize Generator =====================
        if step < FLAGS.initial_step:
          _1, summary = sess.run([initialize_G_model, summary_ini],
                           feed_dict={samsara_gan.x: batch_image[:,:,:,:3], samsara_gan.y: batch_image[:,:,:,3:],
                                      samsara_gan.x_class: x_class, samsara_gan.y_class: y_class})

        # ===================== Main Train =====================
        else:
          start = time.time()
          # get generated images
          fake_y_val, fake_x_val = sess.run([fake_y, fake_x],
                                  feed_dict={samsara_gan.x: batch_image[:,:,:,:3], samsara_gan.y: batch_image[:,:,:,3:],
                                             samsara_gan.x_class: x_class, samsara_gan.y_class: y_class})

          # train
          _, G_loss_val, D_Y_loss_val, D_X_loss_val, summary = (
                sess.run([optimizers, G_loss, D_Y_loss, D_X_loss, summary_op],
                feed_dict={samsara_gan.fake_y: fake_Y_pool.query(fake_y_val),
                           samsara_gan.fake_x: fake_X_pool.query(fake_x_val),
                           samsara_gan.x: batch_image[:,:,:,:3],
                           samsara_gan.y: batch_image[:,:,:,3:],
                           samsara_gan.x_class: x_class, samsara_gan.y_class: y_class}
                ))

          end = time.time()
          current_time = datetime.now().strftime("%Y/%m/%d-%H:%M:%S.%f")
          if step % 100 == 0:
            print('----------- Step %d: -------------' % step)
            print('  G_loss   : {}'.format(G_loss_val))
            print('  D_X_loss : {}'.format(D_X_loss_val))
            print('  D_Y_loss : {}'.format(D_Y_loss_val))
            print('  Duration : {}'.format(end-start))
            print('  Time     : {}'.format(current_time))

        train_writer.add_summary(summary, step)
        train_writer.flush()

        step += 1

        if step % 1000 == 0 and step > int(FLAGS.iteration/100):
          save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
          logging.info(" [*] Model saved in file: %s" % save_path)
          test_model(sess, samsara_gan, step, FLAGS.dataset_dir, checkpoints_dir, FLAGS.image_size, FLAGS.batch_size, x_class, y_class)
          logging.info(' [*] Test model')

    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()

    except Exception as e:
      coord.request_stop(e)

    finally:
      save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
      logging.info(" [*] Model saved in file: %s" % save_path)
      test_model(sess, samsara_gan, step, FLAGS.dataset_dir, checkpoints_dir, FLAGS.image_size, FLAGS.batch_size, x_class, y_class)
      logging.info(' [*] Test model')

      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

def main(unused_argv):
  train()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()

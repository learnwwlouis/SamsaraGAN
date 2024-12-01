import os
from glob import glob
import numpy as np
import scipy.misc

def imread(path, is_grayscale = False):
  if (is_grayscale):
      return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
      return scipy.misc.imread(path, mode='RGB').astype(np.float)

def load_test_data(image_path, fine_size):
  img = imread(image_path)
  img = scipy.misc.imresize(img, [fine_size, fine_size], interp='bicubic')
  img = img/127.5 - 1
  return img

def inverse_transform(images):
  return (images+1.)/2.

def imsave(images, path):
  return scipy.misc.imsave(path, images)

def save_images(images, image_path):
  return imsave(inverse_transform(images), image_path)

def test_model(sess, cycle_gan, counter, dataset_dir, checkpoints_dir, image_size, batch_size, x_class, y_class):
# Saving image in filedir
  rootdir = checkpoints_dir+'/'+dataset_dir+'_imsize_'+str(image_size)+'_batchsize_'+str(batch_size)
  if not os.path.exists(rootdir):
    os.makedirs(rootdir)

  filedir = rootdir + '/'+str(counter)
  if not os.path.exists(filedir):
    os.makedirs(filedir)

  filedir_train = rootdir + '/'+'training'
  if not os.path.exists(filedir_train):
    os.makedirs(filedir_train)

  Test_X  = glob('../data/{}/*.jpg'.format(dataset_dir+'/testA'))
  Test_Y  = glob('../data/{}/*.jpg'.format(dataset_dir+'/testB'))

  Train_X = glob('../data/{}/*.jpg'.format(dataset_dir+'/trainA'))
  Train_Y = glob('../data/{}/*.jpg'.format(dataset_dir+'/trainB'))
  np.random.shuffle(Train_X)
  np.random.shuffle(Train_Y)

  # ============================== for sample_file in Test_X: ==============================
  for sample_file in Test_X:
    sample_image = [load_test_data(sample_file,fine_size=image_size)]
    sample_image = np.array(sample_image).astype(np.float32)

    real_test_x, fake_test_y_mask, fake_test_y, test_reco_x_mask, test_reco_x = sess.run( \
    [cycle_gan.test_x          , cycle_gan.test_fake_y_mask, cycle_gan.test_fake_y,\
     cycle_gan.test_reco_x_mask, cycle_gan.test_reco_x], \
    feed_dict={cycle_gan.test_x: sample_image,
               cycle_gan.x_class: x_class, cycle_gan.y_class: y_class})

    filename = sample_file.split("/",5)[4].split(".",2)[0]
    save_images(real_test_x[0],      './{}/{:06d}_{}_A_real.png'.format(filedir, counter, filename))
    save_images(fake_test_y_mask[0], './{}/{:06d}_{}_B_fake_mask.png'.format(filedir, counter, filename))
    save_images(fake_test_y[0],      './{}/{:06d}_{}_B_fake.png'.format(filedir, counter, filename))
    save_images(test_reco_x_mask[0], './{}/{:06d}_{}_A_reco_mask.png'.format(filedir, counter, filename))
    save_images(test_reco_x[0],      './{}/{:06d}_{}_A_reco.png'.format(filedir, counter, filename))

  # ============================== for sample_file in Test_Y: ==============================
  for sample_file in Test_Y:
    sample_image = [load_test_data(sample_file,fine_size=image_size)]
    sample_image = np.array(sample_image).astype(np.float32)

    test_test_y, fake_test_x_mask, fake_test_x, reco_test_y_mask, reco_test_y = sess.run( \
    [cycle_gan.test_y          , cycle_gan.test_fake_x_mask, cycle_gan.test_fake_x, \
     cycle_gan.test_reco_y_mask, cycle_gan.test_reco_y], \
    feed_dict={cycle_gan.test_y: sample_image,
               cycle_gan.x_class: x_class, cycle_gan.y_class: y_class})

    filename = sample_file.split("/",5)[4].split(".",2)[0]
    save_images(test_test_y[0],      './{}/{:06d}_{}_B_real.png'.format(filedir, counter, filename))
    save_images(fake_test_x_mask[0], './{}/{:06d}_{}_A_fake_mask.png'.format(filedir, counter, filename))
    save_images(fake_test_x[0],      './{}/{:06d}_{}_A_fake.png'.format(filedir, counter, filename))
    save_images(reco_test_y_mask[0], './{}/{:06d}_{}_B_reco_mask.png'.format(filedir, counter, filename))
    save_images(reco_test_y[0],      './{}/{:06d}_{}_B_reco.png'.format(filedir, counter, filename))

  # ============================== for sample_file in Train_X: ==============================
  sample_image = [load_test_data(Train_X[0],fine_size=image_size)]
  sample_image = np.array(sample_image).astype(np.float32)
  test_test_x, fake_test_y_mask, fake_test_y, reco_test_x_mask, reco_test_x = sess.run( \
    [cycle_gan.test_x          , cycle_gan.test_fake_y_mask, cycle_gan.test_fake_y,\
     cycle_gan.test_reco_x_mask, cycle_gan.test_reco_x], \
  feed_dict={cycle_gan.test_x: sample_image,
             cycle_gan.x_class: x_class, cycle_gan.y_class: y_class})

  filename = Train_X[0].split("/",5)[4].split(".",2)[0]
  save_images(test_test_x[0],      './{}/{:06d}_{}_A_real.png'.format(filedir_train, counter, filename))
  save_images(fake_test_y_mask[0], './{}/{:06d}_{}_B_fake_mask.png'.format(filedir_train, counter, filename))
  save_images(fake_test_y[0],      './{}/{:06d}_{}_B_fake.png'.format(filedir_train, counter, filename))
  save_images(reco_test_x_mask[0], './{}/{:06d}_{}_A_reco_mask.png'.format(filedir_train, counter, filename))
  save_images(reco_test_x[0],      './{}/{:06d}_{}_A_reco.png'.format(filedir_train, counter, filename))

  # ============================== for sample_file in Train_Y: ==============================
  sample_image = [load_test_data(Train_Y[0],fine_size=image_size)]
  sample_image = np.array(sample_image).astype(np.float32)
  test_test_y, fake_test_x_mask, fake_test_x, reco_test_y_mask, reco_test_y = sess.run( \
  [cycle_gan.test_y          , cycle_gan.test_fake_x_mask, cycle_gan.test_fake_x, \
   cycle_gan.test_reco_y_mask, cycle_gan.test_reco_y], \
   feed_dict={cycle_gan.test_y: sample_image,
              cycle_gan.x_class: x_class, cycle_gan.y_class: y_class})

  filename = Train_Y[0].split("/",5)[4].split(".",2)[0]
  save_images(test_test_y[0],      './{}/{:06d}_{}_B_real.png'.format(filedir_train, counter, filename))
  save_images(fake_test_x_mask[0], './{}/{:06d}_{}_A_fake_mask.png'.format(filedir_train, counter, filename))
  save_images(fake_test_x[0],      './{}/{:06d}_{}_A_fake.png'.format(filedir_train, counter, filename))
  save_images(reco_test_y_mask[0], './{}/{:06d}_{}_B_reco_mask.png'.format(filedir_train, counter, filename))
  save_images(reco_test_y[0],      './{}/{:06d}_{}_B_reco.png'.format(filedir_train, counter, filename))

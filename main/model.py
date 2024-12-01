import tensorflow as tf
import ops
import utils
from discriminator import Discriminator
from generator import Generator

# REAL_LABEL = 0.9

class SamsaraGAN:
  def __init__(self,
               batch_size=1,
               image_size=256,
               use_lsgan=True,
               norm='instance',
               lambda1=10.0,
               lambda2=10.0,
               learning_rate=2e-4,
               beta1=0.5,
               ngf=32,
               class_num=2
              ):
    """
    Args:
      batch_size: integer, batch size
      image_size: integer, image size
      use_lsgan: boolean
      norm: 'instance' or 'batch'
      lambda1: integer, weight for forward cycle loss (X->Y->X)
      lambda2: integer, weight for backward cycle loss (Y->X->Y)
      learning_rate: float, initial learning rate for Adam
      beta1: float, momentum term of Adam
      ngf: number of gen filters in the first conv layer
      class_num: number of class
    """
    self.lambda1 = lambda1
    self.lambda2 = lambda2
    self.use_lsgan = use_lsgan  # not used
    use_sigmoid    = not use_lsgan  # not used
    self.batch_size = batch_size
    self.image_size = image_size
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.class_num = class_num

    self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

    self.G = Generator('G', self.is_training, ngf=ngf, norm=norm, image_size=image_size, batch_size=batch_size, class_num=class_num)

    self.D_X = Discriminator('D_X', self.is_training, norm=norm, use_sigmoid=use_sigmoid)
    self.D_Y = Discriminator('D_Y', self.is_training, norm=norm, use_sigmoid=use_sigmoid)

    self.x       = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3], name='x')
    self.y       = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3], name='y')

    self.fake_x  = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3], name='fake_x') # previous generated images
    self.fake_y  = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3], name='fake_y') # previous generated images

    self.test_x  = tf.placeholder(tf.float32, shape=[1, image_size, image_size, 3], name='test_x')
    self.test_y  = tf.placeholder(tf.float32, shape=[1, image_size, image_size, 3], name='test_y')

    self.x_class = tf.placeholder(tf.float32, shape=[batch_size, class_num], name='x_class')
    self.y_class = tf.placeholder(tf.float32, shape=[batch_size, class_num], name='y_class')

  def initialize_G_model(self):
    # X -> Y
    fake_y, fake_y_mask = self.G(self.x, self.y_class)
    reco_x, reco_x_mask = self.G(fake_y, self.x_class)

    # Y -> X
    fake_x, fake_x_mask = self.G(self.y, self.x_class)
    reco_y, reco_y_mask = self.G(fake_x, self.y_class)

    cycle_loss = self.cycle_consistency_loss(reco_x, reco_y, self.x, self.y)
    unifo_loss = self.uniform_loss(fake_y_mask) + self.uniform_loss(fake_x_mask) + self.uniform_loss(reco_x_mask) + self.uniform_loss(reco_y_mask)

    pre_g_loss = cycle_loss + unifo_loss

    self.initialize_G = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).\
                                    minimize(pre_g_loss, var_list=self.G.variables)

    _1_1 = tf.summary.scalar('initialize_loss/1_cycle'      , cycle_loss)
    _1_2 = tf.summary.scalar('initialize_loss/2_unifo'      , unifo_loss)
    _1_3 = tf.summary.scalar('initialize_loss/3_pre_g_loss' , pre_g_loss)

    _2_1 = tf.summary.image('initialize_X/1_input',               utils.batch_convert2int(self.x))
    _2_2 = tf.summary.image('initialize_X/2_generated_mask',      utils.batch_convert2int(fake_y_mask))
    _2_3 = tf.summary.image('initialize_X/3_generated',           utils.batch_convert2int(fake_y))
    _2_4 = tf.summary.image('initialize_X/4_reconstruction_mask', utils.batch_convert2int(reco_x_mask))
    _2_5 = tf.summary.image('initialize_X/5_reconstruction',      utils.batch_convert2int(reco_x))

    _3_1 = tf.summary.image('initialize_Y/1_input',               utils.batch_convert2int(self.y))
    _3_2 = tf.summary.image('initialize_Y/2_generated_mask',      utils.batch_convert2int(fake_x_mask))
    _3_3 = tf.summary.image('initialize_Y/3_generated',           utils.batch_convert2int(fake_x))
    _3_4 = tf.summary.image('initialize_Y/4_reconstruction_mask', utils.batch_convert2int(reco_y_mask))
    _3_5 = tf.summary.image('initialize_Y/5_reconstruction',      utils.batch_convert2int(reco_y))

    return (self.initialize_G),\
           (_1_1,_1_2,_1_3,\
            _2_1, _2_2, _2_3, _2_4, _2_5,\
            _3_1, _3_2, _3_3, _3_4, _3_5)

  def model(self):
    # X -> Y
    fake_y, fake_y_mask = self.G(self.x, self.y_class)
    reco_x, reco_x_mask = self.G(fake_y, self.x_class)
    G_gan_loss = self.generator_loss(self.D_Y, self.y, fake_y, fake_y_mask, reco_x_mask, self.y_class)
    D_Y_loss   = self.discriminator_loss(self.D_Y, self.y, self.fake_y, self.y_class, self.x_class)

    # Y -> X
    fake_x, fake_x_mask = self.G(self.y, self.x_class)
    reco_y, reco_y_mask = self.G(fake_x, self.y_class)
    F_gan_loss = self.generator_loss(self.D_X, self.x, fake_x, fake_x_mask, reco_y_mask, self.x_class)
    D_X_loss   = self.discriminator_loss(self.D_X, self.x, self.fake_x, self.x_class, self.y_class)

    cycle_loss = self.cycle_consistency_loss(reco_x, reco_y, self.x, self.y)

    G_loss = G_gan_loss + F_gan_loss + cycle_loss

    # ============================ Test Model ============================
    # X -> Y
    self.test_fake_y, self.test_fake_y_mask = self.G(self.test_x, self.y_class)
    self.test_reco_x, self.test_reco_x_mask = self.G(self.test_fake_y, self.x_class)

    # Y -> X
    self.test_fake_x, self.test_fake_x_mask = self.G(self.test_y, self.x_class)
    self.test_reco_y, self.test_reco_y_mask = self.G(self.test_fake_x, self.y_class)

    _1_1 = tf.summary.scalar('loss/1_G',     G_loss)
    _1_2 = tf.summary.scalar('loss/2_G_1',   G_gan_loss)
    _1_3 = tf.summary.scalar('loss/3_G_2',   F_gan_loss)
    _1_4 = tf.summary.scalar('loss/4_cycle', cycle_loss)
    _1_5 = tf.summary.scalar('loss/5_D_X',   D_X_loss)
    _1_6 = tf.summary.scalar('loss/6_D_Y',   D_Y_loss)

    _2_1 = tf.summary.image('X/1_input',               utils.batch_convert2int(self.x))
    _2_2 = tf.summary.image('X/2_generated_mask',      utils.batch_convert2int(fake_y_mask))
    _2_3 = tf.summary.image('X/3_generated',           utils.batch_convert2int(fake_y))
    _2_4 = tf.summary.image('X/4_reconstruction_mask', utils.batch_convert2int(reco_x_mask))
    _2_5 = tf.summary.image('X/5_reconstruction',      utils.batch_convert2int(reco_x))

    _3_1 = tf.summary.image('Y/1_input',               utils.batch_convert2int(self.y))
    _3_2 = tf.summary.image('Y/2_generated_mask',      utils.batch_convert2int(fake_x_mask))
    _3_3 = tf.summary.image('Y/3_generated',           utils.batch_convert2int(fake_x))
    _3_4 = tf.summary.image('Y/4_reconstruction_mask', utils.batch_convert2int(reco_y_mask))
    _3_5 = tf.summary.image('Y/5_reconstruction',      utils.batch_convert2int(reco_y))

    return G_loss, D_Y_loss, D_X_loss, fake_y, fake_x,\
           (_1_1,_1_2,_1_3,_1_4,_1_5,_1_6,
            _2_1,_2_2,_2_3,_2_4,_2_5,
            _3_1,_3_2,_3_3,_3_4,_3_5)

  def optimize(self, G_loss, D_Y_loss, D_X_loss):
    def make_optimizer(loss, variables, name='Adam'):
      """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
      """
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = self.learning_rate
      end_learning_rate = 0.0
      start_decay_step = 100000
      decay_steps = 100000
      beta1 = self.beta1
      learning_rate = (
                        tf.where(
                                tf.greater_equal(global_step, start_decay_step),
                                tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                                          decay_steps, end_learning_rate,
                                                          power=1.0),
                                                          starter_learning_rate
                                )
                      )
      tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

      learning_step = (
          tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                  .minimize(loss, global_step=global_step, var_list=variables)
      )
      return learning_step

    G_optimizer   = make_optimizer(G_loss,   self.G.variables,   name='Adam_G')
    D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')
    D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')

    with tf.control_dependencies([G_optimizer, D_X_optimizer, D_Y_optimizer]):
      return tf.no_op(name='optimizers')

  def discriminator_loss(self, D, y, fake_y, real_y_class, real_x_class):
    
    y_real      = D(y)
    fake_y_real = D(fake_y)

    # use mean squared error
    error_real = self.feature_loss(y_real,      real_y_class)
    error_fake = self.feature_loss(fake_y_real, real_x_class)

    loss = error_real + error_fake
    return loss

  def generator_loss(self, D, _real, _fake, fake_mask, reco_mask, _class):
    """  fool discriminator into believing that G(x) is real
    """
    fake_p = D(_fake)

    error_fake = self.feature_loss(fake_p, _class)
    error_unif = self.uniform_loss(fake_mask) + self.uniform_loss(reco_mask)

    loss = error_fake + error_unif
    return loss

  def cycle_consistency_loss(self, reco_x, reco_y, x, y):
    """ cycle consistency loss (L1 norm)
    """
    forward_loss  = tf.reduce_mean(tf.abs(reco_x - x))
    backward_loss = tf.reduce_mean(tf.abs(reco_y - y))

    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss

  def feature_loss(self, predict_class, real_class):
    """ GAN loss (Least Squared loss)
    """
    error = 0.0
    count = real_class.get_shape()[-1] # num of classes    
    for k in range(count):
      p = tf.reduce_mean(tf.squared_difference(predict_class[:,:,:,k], real_class[:,k]))
      error = error + p
    count = tf.cast(count, tf.float32)
    error = error / count
    return error

  def get_var(self, x):
    x = tf.reshape(x, shape=[-1])
    m = 0
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared)

  def get_std(self, x):
    return tf.sqrt(self.get_var(x))

  def uniform_loss(self, x):
    error = 0.0
    channel = x.get_shape()[-1] # num of channel
    # channel-wise
    for k in range(channel):
      error = error + self.get_std(x[:,:,:,k])
    channel = tf.cast(channel, tf.float32)
    error = error / channel
    return error

import tensorflow as tf
import ops
import utils
import densenet

class Generator:
  def __init__(self, name, is_training, ngf=32, norm='instance', image_size=256, batch_size=1, class_num=1):
    self.name  = name
    self.reuse = False
    self.ngf   = ngf
    self.norm  = norm
    self.is_training = is_training
    self.image_size  = image_size
    self.batch_size  = batch_size
    self.class_num   = class_num

  def __call__(self, input, label):
    """
    Args:
      input: batch_size x width x height x 3
    Returns:
      output: same size as input
    """

    with tf.variable_scope(self.name):
      # conv layers
      c7s1_32 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='c7s1_32')                             # (?, w, h, 32)

      d64 = ops.dk(c7s1_32, 2*self.ngf, is_training=self.is_training, norm=self.norm,
            reuse=self.reuse, name='d64')                                 # (?, w/2, h/2, 64)

      d128 = ops.dk(d64, 4*self.ngf, is_training=self.is_training, norm=self.norm,
             reuse=self.reuse, name='d128')                               # (?, w/4, h/4, 128)

      res_output = densenet.densenet(d128, label=label, is_training=self.is_training, reuse=self.reuse, name='densenet')

      # fractional-strided convolution
      u64 = ops.uk(res_output, 2*self.ngf, is_training=self.is_training, norm=self.norm,
                   reuse=self.reuse, name='u64')                                  # (?, w/2, h/2, 64)

      u32 = ops.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm,
                   reuse=self.reuse, name='u32', output_size=self.image_size)     # (?, w, h, 32)

      # conv layer
      # Note: the paper said that ReLU and _norm were used
      # but actually tanh was used and no _norm here
      mask = ops.c7s1_k(u32, 3, norm=None,
                        activation=None, reuse=self.reuse, name='mask')           # (?, w, h, 3)

      output = tf.nn.tanh(mask+input)
      mask   = tf.nn.tanh(mask)

    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output, mask

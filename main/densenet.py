import numpy as np
import tensorflow as tf

# ===== Paremeter Settings - number of layers and blocks =====
depth = 40
num_block = 3
N = int((depth-4) // num_block)
grow_rate = 12

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()

  ones = tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , 1])

  for k in range(y_shapes[-1]):
    x = tf.concat([x, y[:,k]*ones], 3)

  return x

def instance_norm(input, name="instance_norm"):
  with tf.variable_scope(name):
    depth  = input.get_shape()[3]
    scale  = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
    offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
        
    return scale*normalized + offset


# ========== Convolution layer ==========
def _conv2d(input, in_feats, out_feats, kernel_size, name="conv"):
  with tf.variable_scope(name) as scope:
    W = tf.get_variable(name="weights", initializer=tf.truncated_normal(shape=[kernel_size, kernel_size, in_feats, out_feats], stddev=np.sqrt(2.0/9/in_feats)))
    conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
  
    return conv

# ========== Dense Connections (block) ==========
def _Hl(name, input, kernel_size, is_training, drop_prob, prev):
  in_feats = input.get_shape().as_list()[3]
  out = input
  with tf.variable_scope(name) as scope:
    out = _conv2d(out, in_feats, 4*grow_rate, kernel_size, name="bottle_conv")
    out = instance_norm(out, name+'_bottle_BN')
    out = tf.nn.relu(out)

    out = _conv2d(out, 4*grow_rate, grow_rate, kernel_size, name="conv")
    out = instance_norm(out, name+'_BN')
    out = tf.nn.relu(out)

    out = tf.concat([out, prev], 3)
  
    return out

# ========== transition layers ==========
def _Tl(name, input, is_training, drop_prob):
  in_feats = input.get_shape().as_list()[3]
  with tf.variable_scope(name) as scope:
    out = _conv2d(input, in_feats, in_feats, 1, name="conv")
    out = instance_norm(out, name+'_BN')
    out = tf.nn.relu(out)

    return out

# ========== Main Densenet ==========
def densenet(images, label, is_training, reuse=False, name='densenet'):
  """Build Densenet model"""

  out = images

  with tf.variable_scope(name, reuse=reuse) as scope:
    for k in range(num_block):
      with tf.variable_scope('block{}'.format(k+1), reuse=reuse) as scope:
        for i in range(N):
          out = _Hl(name="dense_{}".format(i+1), input=out, kernel_size=3, is_training=is_training, drop_prob=0, prev=out)

          # Append class
          out = conv_cond_concat(out, label)

          out = _Tl(name="trans_{}".format(i+2), input=out, is_training=is_training, drop_prob=0)

    return out

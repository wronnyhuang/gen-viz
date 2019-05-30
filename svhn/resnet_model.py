# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# HEAVILY MODIFIED BY RONNY HUANG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import six
import time
import utils, specreg
from tensorflow.python.training import moving_averages

class ResNet(object):
  """ResNet model."""

  def __init__(self, args, mode):
    """ResNet constructor.

    Args:
      args: arguments
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.relu_leakiness = 0.1
    self.optimizer = 'mom'
    self.use_bottleneck = False

    images = tf.placeholder(name='input/images', dtype=tf.float32, shape=(None, 32, 32, 3))
    labels = tf.placeholder(name='input/labels', dtype=tf.float32, shape=(None, args.num_classes))
    self._images = images
    self.labels = labels
    self.args = args
    self.mode = mode

    # build the necessary probes here
    self.lrn_rate = tf.constant(0, tf.float32)
    self.momentum = tf.constant(.9, tf.float32)
    self.speccoef = tf.constant(0, tf.float32)
    self.projvec_beta = tf.constant(0, dtype=tf.float32)

    # build the graph
    self._extra_train_ops = []
    self.build_graph()

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.train.get_or_create_global_step()

    start = time.time()
    self._build_model()
    self._build_train_op()
    print('Graph built in '+str(time.time()-start)+' sec')
    time.sleep(1)

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    with tf.variable_scope('init'):
      x = self._images
      x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    if self.use_bottleneck:
      res_func = self._bottleneck_residual
      # filters = [16, 64, 128, 256]
    else:
      res_func = self._residual
      filters = [16, 16, 32, 64]
      f = self.args.resnet_width
      filters = [16, 16*f, 32*f, 64*f]
      # Uncomment the following codes to use w28-10 wide residual network.
      # It is more memory efficient than very deep residual network and has
      # comparably good performance.
      # https://arxiv.org/pdf/1605.07146v1.pdf
      # filters = [16, 160, 320, 640]
      # Update args.num_resunits to 4

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in six.moves.range(1, self.args.num_resunits):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in six.moves.range(1, self.args.num_resunits):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in six.moves.range(1, self.args.num_resunits):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, self.relu_leakiness)
      x = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      # logits = self._fully_connected(x, self.args.num_classes) # modified by ronny
      logits = tf.layers.dense(x, self.args.num_classes)
      self.predictions = tf.nn.softmax(logits)

    if self.mode=='train' and self.args.poison:
      # cross entropy only for train
      with tf.variable_scope('xent'): # addedby ronny
        self.dirtyOne = tf.placeholder(name='dirtyOne', dtype=tf.float32, shape=[None, 10])
        self.dirtyNeg = tf.placeholder(name='dirtyNeg', dtype=tf.float32, shape=[None, 10])
        self.dirtyPredictions = self.dirtyOne + self.dirtyNeg * self.predictions
        self.xentPerExample = K.categorical_crossentropy(self.labels, self.dirtyPredictions)
        self.xent = tf.reduce_mean(self.xentPerExample)
    else:
      # cross entropy, only for eval
      with tf.variable_scope('xent'):
        self.xentPerExample = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels)
        self.xent = tf.reduce_mean(self.xentPerExample)

    # add accuracy calculation
    truth = tf.argmax(self.labels, axis=1)
    pred = tf.argmax(self.predictions, axis=1)
    self.precision = tf.reduce_mean(tf.to_float(tf.equal(pred, truth)))

  def _build_train_op(self):
    """Build training specific ops for the graph."""

    if self.mode=='eval':
      # add spectral radius calculations
      specreg._spec(self, self.xentPerExample, True, self.args.nohess, self.args.randvec)
      return
    
    elif self.mode == 'curv':
      specreg._spec(self, self.xentPerExample, True, self.args.nohess, self.args.randvec)
      return

    # build gradients for the regular loss with weight decay but no spectral radius
    trainable_variables = tf.trainable_variables()
    self.weight_norm = tf.global_norm(trainable_variables)
    self.loss_orig = self.xent + self._decay() #+ specreg._spec(self, self.xent)
    tstart = time.time()
    grads = tf.gradients(self.loss_orig, trainable_variables)
    print('Built grads: ' + str(time.time() - tstart))

    # build gradients for spectral radius (long operation)
    gradsSpecList = []
    self.gradsSpecCorr= []
    self.loss = self.loss_orig
    if self.mode=='train' and not self.args.poison and not self.args.nohess:

      # build N computations of eigenvalue gradient, each either diff rand direction
      n_grads_spec = self.args.n_grads_spec if self.args.randvec else 1
      valEagerAccum = 0
      for i in range(n_grads_spec):

        # compute spectral radius
        print('=> Spectral radius graph '+str(i))
        specreg._spec(self, self.xentPerExample, False, self.args.nohess, self.args.randvec)
        valEagerAccum = valEagerAccum + self.valEager

        # total loss for training
        if self.args.randvec:
          loss_spec = self.speccoef * tf.exp( -self.args.specexp * self.valEager )
        else:
          loss_spec = self.speccoef * self.valEager
        self.loss = self.loss + loss_spec / n_grads_spec

        # compute the gradient wrt spectral radius and clip
        tstart = time.time()
        gradsSpec = tf.gradients(loss_spec, trainable_variables)
        gradsSpec, self.grad_norm = tf.clip_by_global_norm(gradsSpec, clip_norm=self.args.max_grad_norm)

        # accumulate gradients piecewise additively
        if i==0: gradsSpecAccum = gradsSpec
        else: gradsSpecAccum = [a + g for a,g in zip(gradsSpecAccum, gradsSpec)]
        print('Built gradSpec:', str(time.time()-tstart))

        # record intragradient correlations
        self.gradsSpecCorr.extend([utils.list2corr(gradsSpec, g) for g in gradsSpecList])
        gradsSpecList = gradsSpecList + [gradsSpec]

      self.valEager = valEagerAccum / n_grads_spec
      grads = [ g + a / n_grads_spec for g, a in zip(grads, gradsSpecAccum) ]

    # build optimizer apply_op
    if self.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, self.momentum)
    apply_op = optimizer.apply_gradients(
      zip(grads, trainable_variables),
      global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.summary.histogram(var.op.name, var)

    self.wdec = tf.add_n(costs)
    return tf.multiply(self.args.weight_decay, self.wdec)

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        # tf.summary.histogram(mean.op.name, mean)
        # tf.summary.histogram(variance.op.name, variance)
      # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                           activate_before_residual=False):
    """Bottleneck residual unit with 3 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('common_bn_relu'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 1, in_filter, out_filter/4, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])

    with tf.variable_scope('sub3'):
      x = self._batch_norm('bn3', x)
      x = self._relu(x, self.relu_leakiness)
      x = self._conv('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
      x += orig_x

    tf.logging.info('image after unit %s', x.get_shape())
    return x

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.args.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

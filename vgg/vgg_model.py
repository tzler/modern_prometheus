from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import tensorflow as tf
import numpy as np
import os, sys


def vgg16(inputs, init='from_file', train=True, norm=False, **kwargs):

    out = {}
    out = inputs 
    m = ConvNetFine(**kwargs)
    dropout = .5 if train else None
    vgg_param_path = '/home/biota/cs375-group4/final_project/weights/vgg_weights.npz'
    
    print('input shape: ', inputs['images'].get_shape().as_list())
    
    with tf.contrib.framework.arg_scope([m.conv], init='xavier',
                                        stddev=.01, bias=0, activation='relu', weight_decay=1e-3):

        # preprocessing here instead of with damian's scripts 
        inputs['images'] = tf.cast(inputs['images'], dtype=tf.float32)
        im_n, im_h, im_w, im_d = inputs['images'].get_shape().as_list()
        inputs['images'] = tf.image.resize_images(inputs['images'], [224, 224])
        print('image size after resizing: ', inputs['images'].get_shape().as_list()) 
                
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            inputs['images'] = inputs['images'] - mean        

        with tf.variable_scope('conv1'):
            out['conv1_1'] = m.conv(64, 3, 1, init=init, init_file=vgg_param_path, in_layer= inputs['images'], 
                                  init_layer_keys={'weight': 'conv1_1_W', 'bias': 'conv1_1_b'}, trainable=False)

        with tf.variable_scope('conv2'):
            out['conv1_2'] = m.conv(64, 3, 1, init=init, init_file=vgg_param_path,
                                  init_layer_keys={'weight': 'conv1_2_W', 'bias': 'conv1_2_b'}, trainable=False)
            out['pool1_2'] = m.pool(2, 2)

        with tf.variable_scope('conv3'):
            out['conv2_1'] = m.conv(128, 3, 1, init=init, init_file=vgg_param_path,
                                  init_layer_keys={'weight': 'conv2_1_W', 'bias': 'conv2_1_b'}, trainable=False)

        with tf.variable_scope('conv4'):
            out['conv2_2'] = m.conv(128, 3, 1, init=init, init_file=vgg_param_path,
                                  init_layer_keys={'weight': 'conv2_2_W', 'bias': 'conv2_2_b'}, trainable=False)
            out['pool2_2'] = m.pool(2, 2)

        with tf.variable_scope('conv5'):
            out['conv3_1'] = m.conv(256, 3, 1, init=init, init_file=vgg_param_path,
                                  init_layer_keys={'weight': 'conv3_1_W', 'bias': 'conv3_1_b'}, trainable=False)
            out['conv1_kernel'] = out['conv3_1']
        
        with tf.variable_scope('conv6'):
            out['conv3_2'] = m.conv(256, 3, 1, init=init, init_file=vgg_param_path,
                                  init_layer_keys={'weight': 'conv3_2_W', 'bias': 'conv3_2_b'}, trainable=False)
        with tf.variable_scope('conv7'):
            out['conv3_3'] = m.conv(256, 3, 1, init=init, init_file=vgg_param_path,
                                  init_layer_keys={'weight': 'conv3_3_W', 'bias': 'conv3_3_b'}, trainable=False)
            out['pool3_3'] = m.pool(2, 2)

    	with tf.variable_scope('conv8'):
	    out['conv4_1'] = m.conv(512, 3, 1, init='from_file', init_file=vgg_param_path, 
                                  init_layer_keys={'weight': 'conv4_1_W', 'bias': 'conv4_1_b'}, trainable=False)
    
	with tf.variable_scope('conv9'):
            out['conv4_2'] = m.conv(512, 3, 1, init='from_file', init_file=vgg_param_path, 
                                  init_layer_keys={'weight': 'conv4_2_W', 'bias': 'conv4_2_b'}, trainable=False)
    
	with tf.variable_scope('conv10'):
            out['conv4_3'] = m.conv(512, 3, 1, init='from_file', init_file=vgg_param_path, 
                                  init_layer_keys={'weight': 'conv4_3_W', 'bias': 'conv4_3_b'}, trainable=False)
            out['pool4_3'] = m.pool(2, 2)

        with tf.variable_scope('conv11'):
            out['conv5_1'] = m.conv(512, 3, 1, init='from_file', init_file=vgg_param_path, 
                                  init_layer_keys={'weight': 'conv5_1_W', 'bias': 'conv5_1_b'}, trainable=False)
    	
	with tf.variable_scope('conv12'):
            out['conv5_2'] = m.conv(512, 3, 1, init='from_file', init_file=vgg_param_path, 
                                  init_layer_keys={'weight': 'conv5_2_W', 'bias': 'conv5_2_b'}, trainable=False)
    
	with tf.variable_scope('conv13'):
            out['conv5_3'] = m.conv(512, 3, 1, init='from_file', init_file=vgg_param_path, 
                                  init_layer_keys={'weight': 'conv5_3_W', 'bias': 'conv5_3_b'}, trainable=False)
            out['pool5_3'] = m.pool(2, 2)

        print('pool5_3 shape: ', out['pool5_3'].get_shape().as_list())

        with tf.variable_scope('fc6'):
            out['fc6'] = m.fc(4096, dropout=dropout, init='from_file', init_file=vgg_param_path, 
                                  init_layer_keys={'weight': 'fc6_W', 'bias': 'fc6_b'}, trainable=False)

    	with tf.variable_scope('fc7'):
            out['fc7'] = m.fc(4096, dropout=dropout, init='from_file', init_file=vgg_param_path, 
                                  init_layer_keys={'weight': 'fc7_W', 'bias': 'fc7_b'}, trainable=False)

	with tf.variable_scope('fc8'):
            out['fc8'] = m.fc(1000, activation=None, dropout=None, init='from_file', 
                                  init_file=vgg_param_path, init_layer_keys={'weight': 'fc8_W', 'bias': 'fc8_b'}, 
                                  trainable=True)
        out['pred'] = out['fc8']

        print('\n\n\n contents of outputs: ', out, '\n\n\n')
        return out, {}


class ConvNetOld(object):
    """Basic implementation of ConvNet class compatible with tfutils.
    """

    def __init__(self, seed=None, **kwargs):
        self.seed = seed
        self.output = None
        self._params = OrderedDict()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        name = tf.get_variable_scope().name
        if name not in self._params:
            self._params[name] = OrderedDict()
        self._params[name][value['type']] = value

    @property
    def graph(self):
        return tf.get_default_graph().as_graph_def()

    def initializer(self, kind='xavier', stddev=.01, init_file=None, init_keys=None):
        if kind == 'xavier':
            init = tf.contrib.layers.xavier_initializer(seed=self.seed)
        elif kind == 'trunc_norm':
            init = tf.truncated_normal_initializer(mean=0, stddev=stddev, seed=self.seed)
        elif kind == 'from_file':
            # If we are initializing a pretrained model from a file, load the key from this file
            # Assumes a numpy .npz object
            # init_keys is going to be a dictionary mapping {'weight': weight_key,'bias':bias_key}
            params = np.load(init_file)
            init = {}
            init['weight'] = params[init_keys['weight']]
            init['bias'] = params[init_keys['bias']]
        elif kind == 'variance':
            init = tf.contrib.layers.variance_scaling_initializer(seed=self.seed)
        else:
            raise ValueError('Please provide an appropriate initialization '
                             'method: xavier or trunc_norm')
        return init

    @tf.contrib.framework.add_arg_scope
    def conv(self,
             out_shape,
             ksize=3,
             stride=1,
             padding='SAME',
             init='xavier',
             stddev=.01,
             bias=1,
             activation='relu',
             weight_decay=None,
             in_layer=None,
             init_file=None,
             init_layer_keys=None):
        if in_layer is None:
            in_layer = self.output
        if weight_decay is None:
            weight_decay = 0.
        in_shape = in_layer.get_shape().as_list()[-1]

        if isinstance(ksize, int):
            ksize1 = ksize
            ksize2 = ksize
        else:
            ksize1, ksize2 = ksize

        if init != 'from_file':
            kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
                                     shape=[ksize1, ksize2, in_shape, out_shape],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='weights')
            biases = tf.get_variable(initializer=tf.constant_initializer(bias),
                                     shape=[out_shape],
                                     dtype=tf.float32,
                                     name='bias')
        else:
            init_dict = self.initializer(init,
                                         init_file=init_file,
                                         init_keys=init_layer_keys)
            kernel = tf.get_variable(initializer=init_dict['weight'],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='weights')
            biases = tf.get_variable(initializer=init_dict['bias'],
                                     dtype=tf.float32,
                                     name='bias')

        conv = tf.nn.conv2d(in_layer, kernel,
                            strides=[1, stride, stride, 1],
                            padding=padding)
        self.output = tf.nn.bias_add(conv, biases, name='conv')
        if activation is not None:
            self.output = self.activation(kind=activation)
        self.params = {'input': in_layer.name,
                       'type': 'conv',
                       'num_filters': out_shape,
                       'stride': stride,
                       'kernel_size': (ksize1, ksize2),
                       'padding': padding,
                       'init': init,
                       'stddev': stddev,
                       'bias': bias,
                       'activation': activation,
                       'weight_decay': weight_decay,
                       'seed': self.seed}
        return self.output

    @tf.contrib.framework.add_arg_scope
    def fc(self,
           out_shape,
           init='xavier',
           stddev=.01,
           bias=1,
           activation='relu',
           weight_decay=None,
           dropout=.5,
           in_layer=None,
           init_file=None,
           init_layer_keys=None):

        if in_layer is None:
            in_layer = self.output
        if weight_decay is None:
            weight_decay = 0.
        resh = tf.reshape(in_layer,
                          [in_layer.get_shape().as_list()[0], -1],
                          name='reshape')
        in_shape = resh.get_shape().as_list()[-1]
        if init != 'from_file':
            kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
                                     shape=[in_shape, out_shape],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='weights')
            biases = tf.get_variable(initializer=tf.constant_initializer(bias),
                                     shape=[out_shape],
                                     dtype=tf.float32,
                                     name='bias')
        else:
            init_dict = self.initializer(init,
                                         init_file=init_file,
                                         init_keys=init_layer_keys)
            kernel = tf.get_variable(initializer=init_dict['weight'],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='weights')
            biases = tf.get_variable(initializer=init_dict['bias'],
                                     dtype=tf.float32,
                                     name='bias')

        fcm = tf.matmul(resh, kernel)
        self.output = tf.nn.bias_add(fcm, biases, name='fc')
        if activation is not None:
            self.activation(kind=activation)
        if dropout is not None:
            self.dropout(dropout=dropout)

        self.params = {'input': in_layer.name,
                       'type': 'fc',
                       'num_filters': out_shape,
                       'init': init,
                       'bias': bias,
                       'stddev': stddev,
                       'activation': activation,
                       'weight_decay': weight_decay,
                       'dropout': dropout,
                       'seed': self.seed}
        return self.output

    @tf.contrib.framework.add_arg_scope
    def norm(self,
             depth_radius=2,
             bias=1,
             alpha=2e-5,
             beta=.75,
             in_layer=None):
        if in_layer is None:
            in_layer = self.output
        self.output = tf.nn.lrn(in_layer,
                                depth_radius=np.float(depth_radius),
                                bias=np.float(bias),
                                alpha=alpha,
                                beta=beta,
                                name='norm')
        self.params = {'input': in_layer.name,
                       'type': 'lrnorm',
                       'depth_radius': depth_radius,
                       'bias': bias,
                       'alpha': alpha,
                       'beta': beta}
        return self.output

    @tf.contrib.framework.add_arg_scope
    def pool(self,
             ksize=3,
             stride=2,
             padding='SAME',
             type='max',
             in_layer=None):
        if in_layer is None:
            in_layer = self.output

        if isinstance(ksize, int):
            ksize1 = ksize
            ksize2 = ksize
        else:
            ksize1, ksize2 = ksize

        if type == 'max':
            self.output = tf.nn.max_pool(in_layer,
                                         ksize=[1, ksize1, ksize2, 1],
                                         strides=[1, stride, stride, 1],
                                         padding=padding,
                                         name='pool')
        elif type == 'avg':
            self.output = tf.nn.avg_pool(in_layer,
                                         ksize=[1, ksize1, ksize2, 1],
                                         strides=[1, stride, stride, 1],
                                         padding=padding,
                                         name='pool')
        else:
            raise NotImplementedError('type: {}'.format(type))
        self.params = {'input': in_layer.name,
                       'type': 'maxpool',
                       'kernel_size': (ksize1, ksize2),
                       'stride': stride,
                       'padding': padding}
        return self.output

    def activation(self, kind='relu', in_layer=None):
        if in_layer is None:
            in_layer = self.output
        if kind == 'relu':
            out = tf.nn.relu(in_layer, name='relu')
        elif kind == 'softplus':
            out = tf.nn.softplus(in_layer, name='softplus')
        else:
            raise ValueError("Activation '{}' not defined".format(kind))
        self.output = out
        return out

    def dropout(self, dropout=.5, in_layer=None):
        if in_layer is None:
            in_layer = self.output
        self.output = tf.nn.dropout(in_layer, dropout, seed=self.seed, name='dropout')
        return self.output

    def subsample(self, num_features, rng=None, in_layer=None):
        if in_layer is None:
            in_layer = self.output
        if rng is None:
            rng = np.random.RandomState(self.seed)
        in_shape = in_layer.get_shape().as_list()[-1]
        mask = np.array([0] * (in_shape - num_features) + [1] * num_features).astype(bool)
        rng.shuffle(mask)
        self.output = tf.transpose(tf.boolean_mask(tf.transpose(in_layer), mask))
        self.output.set_shape(self.output.get_shape().as_list()[:-1] + [num_features])
        return self.output

    def bn(self, train, in_layer=None, decay=0.999, epsilon=1e-3):
        if in_layer is None:
            in_layer = self.output
        in_shape = in_layer.get_shape().as_list()[-1]

        beta = tf.get_variable(initializer=tf.zeros_initializer(), shape=[in_shape], name='beta')
        scale = tf.get_variable(initializer=tf.ones_initializer(), shape=[in_shape], name='scale')
        pop_mean = tf.get_variable(initializer=tf.zeros_initializer(), shape=[in_shape], name='bn_mean', trainable=False)
        pop_var = tf.get_variable(initializer=tf.ones_initializer(), shape=[in_shape], name='bn_var', trainable=False)

        if train:
            batch_mean, batch_var = tf.nn.moments(in_layer, axes=list(range(in_layer.get_shape().ndims - 1)))

            update_pop_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            update_pop_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

            with tf.control_dependencies([update_pop_mean, update_pop_var]):
                self.output = tf.nn.batch_normalization(in_layer, batch_mean, batch_var, beta, scale, epsilon)
        else:
            self.output = tf.nn.batch_normalization(in_layer, pop_mean, pop_var, beta, scale, epsilon)

        self.params = {'input': in_layer.name,
                       'type': 'bn',
                       'decay': decay,
                       'epsilon': epsilon,
                        }

        return self.output
        
class ConvNetFine(ConvNetOld):
    def __init__(self, seed=None, **kwargs):
        super(ConvNetFine, self).__init__(seed, **kwargs)

    @tf.contrib.framework.add_arg_scope
    def conv(self,
             out_shape,
             ksize=3,
             stride=1,
             padding='SAME',
             init='xavier',
             stddev=.01,
             bias=1,
             activation='relu',
             weight_decay=None,
             in_layer=None,
             init_file=None,
             init_layer_keys=None,
             trainable=True):
        if in_layer is None:
            in_layer = self.output
        if weight_decay is None:
            weight_decay = 0.
        in_shape = in_layer.get_shape().as_list()[-1]

        if isinstance(ksize, int):
            ksize1 = ksize
            ksize2 = ksize
        else:
            ksize1, ksize2 = ksize

        if init != 'from_file':
            kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
                                     shape=[ksize1, ksize2, in_shape, out_shape],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='weights',
                                     trainable=trainable)
            biases = tf.get_variable(initializer=tf.constant_initializer(bias),
                                     shape=[out_shape],
                                     dtype=tf.float32,
                                     name='bias',
                                     trainable=trainable)
        else:
            init_dict = self.initializer(init,
                                         init_file=init_file,
                                         init_keys=init_layer_keys)
            kernel = tf.get_variable(initializer=init_dict['weight'],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='weights',
                                     trainable=trainable)
            biases = tf.get_variable(initializer=init_dict['bias'],
                                     dtype=tf.float32,
                                     name='bias',
                                     trainable=trainable)

        conv = tf.nn.conv2d(in_layer, kernel,
                            strides=[1, stride, stride, 1],
                            padding=padding)
        self.output = tf.nn.bias_add(conv, biases, name='conv')
        if activation is not None:
            self.output = self.activation(kind=activation)
        self.params = {'input': in_layer.name,
                       'type': 'conv',
                       'num_filters': out_shape,
                       'stride': stride,
                       'kernel_size': (ksize1, ksize2),
                       'padding': padding,
                       'init': init,
                       'stddev': stddev,
                       'bias': bias,
                       'activation': activation,
                       'weight_decay': weight_decay,
                       'seed': self.seed}
        return self.output

    @tf.contrib.framework.add_arg_scope
    def fc(self,
           out_shape,
           init='xavier',
           stddev=.01,
           bias=1,
           activation='relu',
           weight_decay=None,
           dropout=.5,
           in_layer=None,
           init_file=None,
           init_layer_keys=None,
           trainable=True):

        if in_layer is None:
            in_layer = self.output
        if weight_decay is None:
            weight_decay = 0.
        resh = tf.reshape(in_layer,
                          [in_layer.get_shape().as_list()[0], -1],
                          name='reshape')
        in_shape = resh.get_shape().as_list()[-1]
        if init != 'from_file':
            kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
                                     shape=[in_shape, out_shape],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='weights',
                                     trainable=trainable)
            biases = tf.get_variable(initializer=tf.constant_initializer(bias),
                                     shape=[out_shape],
                                     dtype=tf.float32,
                                     name='bias',
                                     trainable=trainable)
        else:
            init_dict = self.initializer(init,
                                         init_file=init_file,
                                         init_keys=init_layer_keys)
            kernel = tf.get_variable(initializer=init_dict['weight'],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='weights',
                                     trainable=trainable)
            biases = tf.get_variable(initializer=init_dict['bias'],
                                     dtype=tf.float32,
                                     name='bias',
                                     trainable=trainable)

        fcm = tf.matmul(resh, kernel)
        self.output = tf.nn.bias_add(fcm, biases, name='fc')
        if activation is not None:
            self.activation(kind=activation)
        if dropout is not None:
            self.dropout(dropout=dropout)

        self.params = {'input': in_layer.name,
                       'type': 'fc',
                       'num_filters': out_shape,
                       'init': init,
                       'bias': bias,
                       'stddev': stddev,
                       'activation': activation,
                       'weight_decay': weight_decay,
                       'dropout': dropout,
                       'seed': self.seed}
        return self.output


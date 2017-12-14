from __future__ import division, print_function, absolute_import
import os, sys
import inspect
from functools import wraps
from collections import OrderedDict
from contextlib import contextmanager
import numpy as np

import tensorflow as tf

from tfutils import base, data, model, optimizer, utils
from vgg_retina_model import vgg16

import json
import copy

train_net = True #False
stim_type = 'naturalscene'
# Figure out the hostname
host = os.uname()[1]
if train_net:
    print('In train mode...')
    TOTAL_BATCH_SIZE = 5000
    MB_SIZE = 1000
    NUM_GPUS = 1
else:
    print('In val mode...')
    if stim_type == 'whitenoise':
        TOTAL_BATCH_SIZE = 5957
        MB_SIZE = 5957
        NUM_GPUS = 1
    else:
        TOTAL_BATCH_SIZE = 5956
        MB_SIZE = 5956
        NUM_GPUS = 1


if not isinstance(NUM_GPUS, list):
    DEVICES = ['/gpu:' + str(i) for i in range(NUM_GPUS)]
else:
    DEVICES = ['/gpu:' + str(i) for i in range(len(NUM_GPUS))]

MODEL_PREFIX = 'model_0'

# Data parameters
if stim_type == 'whitenoise':
    N_TRAIN = 323762
    N_TEST = 5957
else:
    N_TRAIN = 323756
    N_TEST = 5956

INPUT_BATCH_SIZE = 1024 # queue size
OUTPUT_BATCH_SIZE = TOTAL_BATCH_SIZE
print('TOTAL BATCH SIZE:', OUTPUT_BATCH_SIZE)
NUM_BATCHES_PER_EPOCH = N_TRAIN // OUTPUT_BATCH_SIZE
IMAGE_SIZE_RESIZE = 50

DATA_PATH = '/mnt/data/deepretina_data/tf_records/' + stim_type
print('Data path: ', DATA_PATH)

# data provider
class retinaTF(data.TFRecordsParallelByFileProvider):

  def __init__(self,
               source_dirs,
               resize=IMAGE_SIZE_RESIZE,
               **kwargs
               ):

    if resize is None:
      self.resize = 50
    else:
      self.resize = resize

    postprocess = {'images': [], 'labels': []}
    postprocess['images'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
    postprocess['images'].insert(1, (tf.reshape, ([-1] + [50, 50, 40], ), {}))
    postprocess['images'].insert(2, (self.postproc_imgs, (), {})) 

    postprocess['labels'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
    postprocess['labels'].insert(1, (tf.reshape, ([-1] + [5], ), {}))

    super(retinaTF, self).__init__(
      source_dirs,
      postprocess=postprocess,
      **kwargs
    )


  def postproc_imgs(self, ims):
    def _postprocess_images(im):
        im = tf.image.resize_images(im, [self.resize, self.resize])
        return im
    return tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=tf.float32)

def poisson_loss(logits, labels):
    return tf.reduce_mean(logits - (labels * tf.log(logits + 1e-8)), axis=-1)

def mean_loss_with_reg(loss):
    return tf.reduce_mean(loss) + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res

def loss_metric(inputs, outputs, target, **kwargs):
    metrics_dict = {}
    metrics_dict['poisson_loss'] = mean_loss_with_reg(poisson_loss(logits=outputs, labels=inputs[target]), **kwargs)
    return metrics_dict

def mean_losses_keep_rest(step_results):
    retval = {}
    keys = step_results[0].keys()
    for k in keys:
        plucked = [d[k] for d in step_results]
        if isinstance(k, str) and 'loss' in k:
            retval[k] = np.mean(plucked)
        else:
            retval[k] = plucked
    return retval

def examine_responses(inputs, outputs, target):
    resp_dict = {}
    print(outputs, outputs.shape)
    resp_dict['pred_resp'] = outputs
    resp_dict['gt_resp'] = inputs[target]
    return resp_dict

def eval_responses(target_params = {'func': examine_responses, 'target': 'labels'}, dbname='deepretina', collname=stim_type, expid='trainval1', step=None, prefix='eval0', group='test'):
    params = {}
    params['model_params'] = {'func': vgg16,
                              'target': 'conv2_1',
                              }
    
    if step is None:
        query = None
        stepstr = 'recent'
    else:
        query = {'step': step}
        stepstr = str(step)
    params['load_params'] = {'host': 'localhost', 'port': 28887, 'dbname': dbname, 'collname': collname, 'exp_id': expid, 'do_restore': True, 'query': query}

    params['save_params'] = {'exp_id': prefix + '_step' + stepstr,                              
                              'save_intermediate_freq': 1,
                              'save_to_gfs': ['pred_resp', 'gt_resp']}

    if group == 'train':
        assert(N_TRAIN % OUTPUT_BATCH_SIZE == 0)
        num_steps = N_TRAIN // OUTPUT_BATCH_SIZE
    elif group == 'test':
        assert(N_TEST % OUTPUT_BATCH_SIZE == 0)
        num_steps = N_TEST // OUTPUT_BATCH_SIZE
    filepattern_str = group + '*.tfrecords'

    params['validation_params'] = {prefix: {'data_params': {
            'func': retinaTF,
            'source_dirs': [os.path.join(DATA_PATH, 'images'), os.path.join(DATA_PATH, 'labels')],
            'resize': IMAGE_SIZE_RESIZE,
            'batch_size': INPUT_BATCH_SIZE,
            'file_pattern': filepattern_str,
            'n_threads': 1,
            'shuffle': False
        },
        'queue_params': {
            'queue_type': 'fifo',
            'batch_size': OUTPUT_BATCH_SIZE,
            'capacity': 11*INPUT_BATCH_SIZE,
            'min_after_dequeue': 10*INPUT_BATCH_SIZE,
            'seed': 0,},
         'num_steps': num_steps,
         'targets': target_params,
         'agg_func': mean_losses_keep_rest}}

    base.test_from_params(**params)


# model parameters

default_params = {
    'save_params': {
        'host': 'localhost',
        'port': 28887,
        'dbname': 'deepretina',
        'collname': stim_type,
        'exp_id': 'trainval0',

        'do_save': True,
        'save_initial_filters': True,
        'save_metrics_freq': 200,  # keeps loss from every SAVE_LOSS_FREQ steps.
        'save_valid_freq': 200,
        'save_filters_freq': 200,
        'cache_filters_freq': 200,
        # 'cache_dir': None,  # defaults to '~/.tfutils'
    },

    'load_params': {
        # 'do_restore': False,
        'query': None
    },

    'model_params': {
        'func': vgg16,
        'target': 'conv2_1',
        # 'init': 'xavier,'
    },

    'train_params': {
        'minibatch_size': MB_SIZE,
        'data_params': {
            'func': retinaTF,
            'source_dirs': [os.path.join(DATA_PATH, 'images'), os.path.join(DATA_PATH, 'labels')],
            'resize': IMAGE_SIZE_RESIZE,
            'batch_size': INPUT_BATCH_SIZE,
            'file_pattern': 'train*.tfrecords',
            'n_threads': 4
        },
        'queue_params': {
            'queue_type': 'random',
            'batch_size': OUTPUT_BATCH_SIZE,
            'capacity': 11*INPUT_BATCH_SIZE,
            'min_after_dequeue': 10*INPUT_BATCH_SIZE,
            'seed': 0,
        },
        'thres_loss': float('inf'),
        'num_steps': 200 * NUM_BATCHES_PER_EPOCH,  # number of steps to train
        'validate_first': True,
    },

    'loss_params': {
        'targets': ['labels'],
        'agg_func': mean_loss_with_reg,
        'loss_per_case_func': poisson_loss
    },

    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'learning_rate': 1e-4,
        'decay_rate': 1.0, # constant learning rate
        'decay_steps': NUM_BATCHES_PER_EPOCH,
        'staircase': True
    },

    'optimizer_params': {
        'func': optimizer.ClipOptimizer,
        'optimizer_class': tf.train.AdamOptimizer,
        'clip': True,
        'trainable_names': None
    },

    'validation_params': {
        'test_loss': {
            'data_params': {
                'func': retinaTF,
                'source_dirs': [os.path.join(DATA_PATH, 'images'), os.path.join(DATA_PATH, 'labels')],
                'resize': IMAGE_SIZE_RESIZE,
                'batch_size': INPUT_BATCH_SIZE,
                'file_pattern': 'test*.tfrecords',
                'n_threads': 4
            },
            'targets': {
                'func': loss_metric,
                'target': 'labels',
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': MB_SIZE,
                'capacity': 11*INPUT_BATCH_SIZE,
                'min_after_dequeue': 10*INPUT_BATCH_SIZE,
                'seed': 0,
            },
            'num_steps': N_TEST // MB_SIZE + 1,
            'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': online_agg
        },
        'train_loss': {
            'data_params': {
                'func': retinaTF,
                'source_dirs': [os.path.join(DATA_PATH, 'images'), os.path.join(DATA_PATH, 'labels')],
                'resize': IMAGE_SIZE_RESIZE,
                'batch_size': INPUT_BATCH_SIZE,
                'file_pattern': 'train*.tfrecords',
                'n_threads': 4
            },
            'targets': {
                'func': loss_metric,
                'target': 'labels',
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': MB_SIZE,
                'capacity': 11*INPUT_BATCH_SIZE,
                'min_after_dequeue': 10*INPUT_BATCH_SIZE,
                'seed': 0,
            },
            'num_steps': N_TRAIN // OUTPUT_BATCH_SIZE + 1,
            'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': online_agg
        }

    },
    'log_device_placement': False,  # if variable placement has to be logged
}

def train_nipscnn_ns():
    params = copy.deepcopy(default_params)
    params['save_params']['dbname'] = 'deepretina'
    params['save_params']['collname'] = stim_type
    params['save_params']['exp_id'] = 'trainval0'

    base.get_params()
    base.train_from_params(**params)

 
if __name__ == '__main__':
    train_nipscnn_ns()

    # eval_responses(target_params = {'func': examine_responses, 'target': 'labels'}, 
    #                dbname='deepretina', 
    #                collname=stim_type, 
    #                expid='trainval1', 
    #                step=None, 
    #                prefix='nste1', 
    #                group='test')




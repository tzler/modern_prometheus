"""
runs vgg16 in it's original format on imagenet. 
"""

import os
import numpy as np
import tensorflow as tf
from tfutils import base, data, model, optimizer, utils
from dataprovider import ImageNetDataProvider

# specific to final project 
from vgg_model import vgg16
database_name = 'modern_prometheus'
collection_name = 'vgg_models'
experiment_id = 'vgg16' 


class ImageNetExperiment():
    """
    Defines the ImageNet training experiment
    """
    class Config():
        """
        Holds model hyperparams and data information.
        """
        batch_size = 128
        data_path = '/datasets/TFRecord_Imagenet_standard'
        seed = 0
        crop_size = 224
        thres_loss = 1000
        n_epochs = 1
        train_steps = ImageNetDataProvider.N_TRAIN / batch_size * n_epochs
        val_steps = np.ceil(ImageNetDataProvider.N_VAL / batch_size).astype(int)
        print('train_steps: ', train_steps, 'val_steps: ', val_steps) 
    
    def setup_params(self):
        """
        """
        params = {}

        """
        train_params defines the training parameters 
        """
        params['train_params'] = {
            'data_params': {
                # ImageNet data provider arguments
                'func': ImageNetDataProvider,
                'data_path': self.Config.data_path,
                'group': 'train',
                'crop_size': self.Config.crop_size,
                # TFRecords (super class) data provider arguments
                'file_pattern': 'train*.tfrecords',
                'batch_size': self.Config.batch_size,
                'shuffle': False,
                'shuffle_seed': self.Config.seed,
                'file_grab_func': self.subselect_tfrecords,
                'n_threads': 4,
            },
            'queue_params': {
                'queue_type': 'random',
                'batch_size': self.Config.batch_size,
                'seed': self.Config.seed,
                'capacity': self.Config.batch_size * 10,
                'min_after_dequeue': self.Config.batch_size * 5,
            },
            'targets': {
                'func': self.return_outputs,
                'targets': [],
            },
            'num_steps': self.Config.train_steps,
            'thres_loss': self.Config.thres_loss,
            'validate_first': True, #tyler dec 11th just changed this to from False ,
        }

        """
        validation_params similar to train_params defines the validation parameters.
        """

        params['validation_params'] = {
            'topn_val': {
                'data_params': {
                    # ImageNet data provider arguments
                    'func': ImageNetDataProvider,
                    'data_path': self.Config.data_path,
                    'group': 'val',
                    'crop_size': self.Config.crop_size,
                    # TFRecords (super class) data provider arguments
                    'file_pattern': 'validation*.tfrecords',
                    'batch_size': self.Config.batch_size,
                    'shuffle': False,
                    'shuffle_seed': self.Config.seed,
                    'file_grab_func': self.subselect_tfrecords,
                    'n_threads': 4,
                },
                'queue_params': {
                    'queue_type': 'fifo',
                    'batch_size': self.Config.batch_size,
                    'seed': self.Config.seed,
                    'capacity': self.Config.batch_size * 10,
                    'min_after_dequeue': self.Config.batch_size * 5,
                },
                'targets': {
                    'func': self.in_top_k,
                },
                'num_steps': self.Config.val_steps,
                'agg_func': self.agg_mean, 
                'online_agg_func': self.online_agg_mean,
            }
        }

        """
        model_params defines the model 
        
        """
        params['model_params'] = {
            'func': vgg16, 
        }

        """
        loss_params defines your training loss.
        
        """
        params['loss_params'] = {
            'targets': ['labels'],
            'agg_func': tf.reduce_mean,
            'loss_per_case_func': loss_per_case_func,
            'loss_per_case_func_params' : {'_outputs': 'outputs', 
                '_targets_$all': 'inputs'},
            'loss_func_kwargs' : {},
        }

        """
        learning_rate_params defines the learning rate, decay and learning function.

        """
        
        def learning_rate_params_func(global_step, boundaries, values):
            return tf.train.piecewise_constant(global_step, boundaries, values)

        params['learning_rate_params'] = {
            'func': learning_rate_params_func,
            'boundaries': list(np.array([150000, 300000, 450000]).astype(np.int64)),
            'values': [0.01, 0.005, 0.001, 0.0005]
            #'decay_steps': ImageNetDataProvider.N_TRAIN / self.batch_size,
            #'decay_rate': 0.95,
            #'staircase': True,
        }

        """
        optimizer_params defines the optimizer.

        """
        params['optimizer_params'] = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.MomentumOptimizer,
            'clip': False,
            'momentum': .9,
        }

        """
        save_params defines how, where and when your training results are saved
        in the database.

        """
        params['save_params'] = {
            'host': 'localhost',
            'port': 24444,
            'dbname': database_name,
            'collname': collection_name,
            'exp_id': experiment_id, #+ '_save', 
            'save_valid_freq': 1,
            'save_filters_freq': 1,
            'cache_filters_freq': 1,
            'save_metrics_freq': 1,
            'save_initial_filters' : True, 
            'save_to_gfs': [],
            'do_save': True,
        }

#        """
#        load_params defines how and if a model should be restored from the database.
#
#        """
#        params['load_params'] = {
#            'do_restore': False,
#            'query': None,
#        }

        return params

    def agg_mean(self, x):
        return {k: np.mean(v) for k, v in x.items()}


    def in_top_k(self, inputs, outputs):
        """
        Implements top_k loss for validation

        """
        return {'top1': tf.nn.in_top_k(outputs['pred'], inputs['labels'], 1),
                'top5': tf.nn.in_top_k(outputs['pred'], inputs['labels'], 5)}


    def subselect_tfrecords(self, path):
        """
        Illustrates how to subselect files for training or validation
        """
        all_filenames = os.listdir(path)
        rng = np.random.RandomState(seed=SEED)
        rng.shuffle(all_filenames)
        return [os.path.join(path, fn) for fn in all_filenames
                if fn.endswith('.tfrecords')]


    def return_outputs(self, inputs, outputs, targets, **kwargs):
        """
        Illustrates how to extract desired targets from the model
        """
        retval = {}
        for target in targets:
            retval[target] = outputs[target]
        return retval


    def online_agg_mean(self, agg_res, res, step):
        """
        Appends the mean value for each key
        """
        if agg_res is None:
            agg_res = {k: [] for k in res}
        for k, v in res.items():
            agg_res[k].append(np.mean(v))
        return agg_res


def loss_per_case_func(inputs, outputs):
    labels = outputs['labels']
    logits = outputs['pred']
    print('labels: ', labels.shape)
    print('logits: ', logits.shape)

    # Compute your_loss using tensorflow sparse softmax
    your_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits=logits)

    return your_loss

def learning_rate_func(global_step):
    values= [.01, .005, .001, .0005],
    boundaries= list(np.array([150000, 300000, 450000]).astype(np.int64))
    return tf.train.piecewise_constant(x = global_step, values = values, boundaries = boundaries)

if __name__ == '__main__':
    """
    Illustrates how to run the configured model using tfutils
    """
    base.get_params()
    m = ImageNetExperiment()
    params = m.setup_params()
    base.train_from_params(**params)

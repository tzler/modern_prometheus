"""
Welcom to the second part of the assignment 1! In this section, we will learn
how to analyze our trained model and evaluate its performance on predicting
neural data.
Mainly, you will first learn how to load your trained model from the database
and then how to use tfutils to evaluate your model on neural data using dldata.
The evaluation will be performed using the 'agg_func' in 'validation_params',
which operates on the aggregated validation results obtained from running the
model on the stimulus images. So let's get started!

Note: Although you will only have to edit a small fraction of the code at the
beginning of the assignment by filling in the blank spaces, you will need to 
build on the completed starter code to fully complete the assignment,
We expect that you familiarize yourself with the codebase and learn how to
setup your own experiments taking the assignments as a basis. This code does
not cover all parts of the assignment and only provides a starting point. To
fully complete the assignment significant changes have to be made and new 
functions need to be added after filling in the blanks. Also, for your projects
we won't give out any code and you will have to use what you have learned from
your assignments. So please always carefully read through the entire code and
try to understand it. If you have any questions about the code structure,
we will be happy to answer it.

Attention: All sections that need to be changed to complete the starter code
are marked with EDIT!
"""


from __future__ import division
import os
import numpy as np
import tensorflow as tf
import tabular as tb
import itertools

from scipy.stats import spearmanr
from dldata.metrics.utils import compute_metric_base
from tfutils import base, data, model, optimizer, utils

from utils import post_process_neural_regression_msplit_preprocessed
from dataprovider import NeuralDataProvider
from vgg_model import vgg16


database_name = 'deep_retina'
collection_name = 'vgg'
experiment_id = 'reference_damian_preprocessing'
descriptor = '_conv3'

class NeuralDataExperiment():
    """
    Defines the neural data testing experiment
    """
    class Config():
        """
        Holds model hyperparams and data information.
        The config class is used to store various hyperparameters and dataset
        information parameters. You will need to change the target layers,
        exp_id, and might have to modify 'conv1_kernel' to the name of your
        first layer, once you start working with different models. Set the seed 
        number to your group number. But please do not change the rest. 

        You will have to EDIT this part. Please set your exp_id here.
                         'conv1', 
                         'conv2', 
                         'conv3', 
                         'conv4', 
                         'conv5', 
                         'pool5', 
                         'fc6', 
                         'fc7', 'fc8']
        """
        target_layers = [
# 			'conv1_1',
#                        'conv1_2',
#                        'pool1_2',
#                        'conv2_1',
#                        'conv2_2',
#                        'pool2_2',
                        'conv3_1',
                        'conv3_2',
                        'conv3_3',
                        'pool3_3']
#                        'conv4_1',
#                        'conv4_2',
#                        'conv4_3',
#                        'pool4_3',
#                        'conv5_1',
#                        'conv5_2',
#                        'conv5_3']
#                        'pool5_3',
#                        'fc6',
#                        'fc7',
#                        'fc8']




        
        
        
        extraction_step=3000
        exp_id = experiment_id
        data_path = '/datasets/neural_data/tfrecords_with_meta'
        noise_estimates_path = '/datasets/neural_data/noise_estimates.npy'
        batch_size = 2
        seed = 4
        crop_size = 224
        gfs_targets = [] 
        extraction_targets = [attr[0] for attr in NeuralDataProvider.ATTRIBUTES] \
            + target_layers
        assert NeuralDataProvider.N_VAL % batch_size == 0, \
                ('number of examples not divisible by batch size!')
        val_steps = int(NeuralDataProvider.N_VAL / batch_size)


    def setup_params(self):
        """
        This function illustrates how to setup up the parameters for train_from_params
        """
        params = {}

        """
        validation_params similar to train_params defines the validation parameters.
        It has the same arguments as train_params and additionally
            agg_func: function that aggregates the validation results across batches,
                e.g. to calculate the mean of across batch losses
            online_agg_func: function that aggregates the validation results across
                batches in an online manner, e.g. to calculate the RUNNING mean across
                batch losses

        Note: Note how we switched the data provider from the ImageNetDataProvider 
        to the NeuralDataProvider since we are now working with the neural data.
        """
        params['validation_params'] = {
            'valid0': {
                'data_params': {
                    # ImageNet data provider arguments
                    'func': NeuralDataProvider,
                    'data_path': self.Config.data_path,
                    'crop_size': self.Config.crop_size,
                    # TFRecords (super class) data provider arguments
                    'file_pattern': '*.tfrecords',
                    'batch_size': self.Config.batch_size,
                    'shuffle': False,
                    'shuffle_seed': self.Config.seed, 
                    'n_threads': 1,
                },
                'queue_params': {
                    'queue_type': 'fifo',
                    'batch_size': self.Config.batch_size,
                    'seed': self.Config.seed,
                    'capacity': self.Config.batch_size * 10,
                    'min_after_dequeue': self.Config.batch_size * 1,
                },
                'targets': {
                    'func': self.return_outputs,
                    'targets': self.Config.extraction_targets,
                },
                'num_steps': self.Config.val_steps,
                'agg_func': self.neural_analysis,
                'online_agg_func': self.online_agg,
            }
        }

        """
        model_params defines the model i.e. the architecture that 
        takes the output of the data provider as input and outputs 
        the prediction of the model.

        You will need to EDIT this part. Switch out the model 'func' as 
        needed when running experiments on different models. The default
        is set to the alexnet model you implemented in the first part of the
        assignment.
        """
        params['model_params'] = {
            'func': vgg16,
        }

        """
        save_params defines how, where and when your training results are saved
        in the database.

        You will need to EDIT this part. Set your own 'host' ('localhost' if local,
        mongodb IP if remote mongodb), 'port', 'dbname', and 'collname' if you want
        to evaluate on a different model than the pretrained alexnet model.
        'exp_id' has to be set in Config.
        """
        params['save_params'] = {
            'host': 'localhost',
            'port': 24444,
            'dbname': database_name,
            'collname': collection_name, 
            'exp_id': experiment_id + descriptor,
            'save_to_gfs': self.Config.gfs_targets,
        }

        """
        load_params defines how and if a model should be restored from the database.

        You will need to EDIT this part. Set your own 'host' ('localhost' if local,
        mongodb IP if remote mongodb), 'port', 'dbname', and 'collname' if you want
        to evaluate on a different model than the pretrained alexnet model.
        'exp_id' has to be set in Config.
        """
        params['load_params'] = {
            'host': 'localhost',
            'port': 24444,
            'dbname': database_name,
            'collname': collection_name, 
            'exp_id': experiment_id,
            'do_restore': True,
            'query': {'step': self.Config.extraction_step} \
                    if self.Config.extraction_step is not None else None,
        }

        params['inter_op_parallelism_threads'] = 500

        return params


    def return_outputs(self, inputs, outputs, targets, **kwargs):
        """
        Illustrates how to extract desired targets from the model
        """
        retval = {}
        for target in targets:
            retval[target] = outputs[target]
        return retval


    def online_agg(self, agg_res, res, step):
        """
        Appends the value for each key
        """
        if agg_res is None:
            agg_res = {k: [] for k in res}
        for k, v in res.items():
            if 'kernel' in k:
                agg_res[k] = v
            else:
                agg_res[k].append(v)
        return agg_res


    def parse_meta_data(self, results):
        """
        Parses the meta data from tfrecords into a tabarray
        """
        meta_keys = [attr[0] for attr in NeuralDataProvider.ATTRIBUTES \
                if attr[0] not in ['images', 'it_feats']]
        meta = {}
        for k in meta_keys:
            if k not in results:
                raise KeyError('Attribute %s not loaded' % k)
            meta[k] = np.concatenate(results[k], axis=0)
        return tb.tabarray(columns=[list(meta[k]) for k in meta_keys], names = meta_keys)


    def categorization_test(self, features, meta, data_subset):
        """
        Performs a categorization test using dldata

        You will need to EDIT this part. Define the specification to
        do a categorization on the neural stimuli using 
        compute_metric_base from dldata.
        """
        print('Categorization test...')
        category_eval_spec = {
            'npc_train': None,
            'npc_test': 2,
            'num_splits': 20,
            'npc_validate': 0,
            'metric_screen': 'classifier',
            'metric_labels': None,
            'metric_kwargs': {'model_type': 'svm.LinearSVC',
                              'model_kwargs': {'C': 5e-3},
                             },
            'labelfunc': 'category',
            'train_q': {'var': data_subset},
            'test_q': {'var': data_subset},
            'split_by': 'obj'
        }
        res = compute_metric_base(features, meta, category_eval_spec)
        res.pop('split_results')
        return res
    
    def within_categorization_test(self, features, meta, data_subset, category):
        """
        Performs a within-category categorization test using dldata

        You will need to EDIT this part. Define the specification to
        do within category categorization on the neural stimuli using 
        compute_metric_base from dldata.
        """
        print('Within-Category Categorization test...')
        category_eval_spec = {
            'npc_train': None,
            'npc_test': 2,
            'num_splits': 20,
            'npc_validate': 0,
            'metric_screen': 'classifier',
            'metric_labels': None,
            'metric_kwargs': {'model_type': 'svm.LinearSVC',
                              'model_kwargs': {'C': 5e-3},
                             },
            'labelfunc': 'obj',
            'train_q': {'var': data_subset, 'category': category},
            'test_q': {'var': data_subset, 'category': category},
            'split_by': 'obj'
        }
        res = compute_metric_base(features, meta, category_eval_spec)
        res.pop('split_results')
        return res

    def pos_regression_test(self, features, meta, data_subset):
        """
        Illustrates how to perform a regression test using dldata

        You will need to EDIT this part. Define the specification to
        do a regression on the IT neurons using compute_metric_base from dldata.
        """
        print('Position regression test...')
        pos_reg_eval_spec = {
            'labelfunc': 'ty',
            'metric_kwargs': {'model_type': 'linear_model.RidgeCV'},
            'metric_labels': None,
            'metric_screen': 'regression',
            'npc_test': 2,
            'npc_train': None,
            'npc_validate': 0,
            'num_splits': 20,
            'split_by': 'obj',
            'test_q': {'var': data_subset},
            'train_q': {'var': data_subset}
        }
        res = compute_metric_base(features, meta, pos_reg_eval_spec)
        res.pop('split_results')
        return res

    def regression_test(self, features, IT_features, meta, data_subset):
        """
        Illustrates how to perform a regression test using dldata

        You will need to EDIT this part. Define the specification to
        do a regression on the IT neurons using compute_metric_base from dldata.
        """
        print('Regression test...')
        it_reg_eval_spec = {
            'labelfunc': lambda x: (IT_features, None),
            'metric_kwargs': {'model_kwargs': {'n_components': 25, 'scale': False},
                              'model_type': 'pls.PLSRegression'},
            'metric_labels': None,
            'metric_screen': 'regression',
            'npc_test': 10,
            'npc_train': 70,
            'npc_validate': 0,
            'num_splits': 5,
            'split_by': 'obj',
            'test_q': {'var': data_subset},
            'train_q': {'var': data_subset}
        }
        res = compute_metric_base(features, meta, it_reg_eval_spec)
        espec = (('all','','IT_regression'), it_reg_eval_spec)
        post_process_neural_regression_msplit_preprocessed(
                res, self.Config.noise_estimates_path)
        res.pop('split_results')
        return res


    def compute_rdm(self, features, meta, mean_objects=False):
        """
        Computes the RDM of the input features

        You will need to EDIT this part. Compute the RDM of features which is a
        [N_IMAGES x N_FEATURES] matrix. The features are then averaged across
        images of the same category which creates a [N_CATEGORIES x N_FEATURES]
        matrix that you have to work with.
        """
        print('Computing RDM...')
        if mean_objects:
            object_list = list(itertools.chain(
                *[np.unique(meta[meta['category'] == c]['obj']) \
                        for c in np.unique(meta['category'])]))
            features = np.array([features[(meta['obj'] == o.rstrip('_'))].mean(0) \
                    for o in object_list])
        rdm = 1 - np.corrcoef(features)
        return rdm


    def get_features(self, results, num_subsampled_features=None):
        """
        Extracts, preprocesses and subsamples the target features
        and the IT features
        """
        features = {}
        for layer in self.Config.target_layers:
            feats = np.concatenate(results[layer], axis=0)
            feats = np.reshape(feats, [feats.shape[0], -1])
            if num_subsampled_features is not None:
                features[layer] = \
                        feats[:, np.random.RandomState(0).permutation(
                            feats.shape[1])[:num_subsampled_features]]

        IT_feats = np.concatenate(results['it_feats'], axis=0)

        return features, IT_feats


    def neural_analysis(self, results):
        """
        Performs an analysis of the results from the model on the neural data.
        This analysis includes:
            - saving the conv1 kernels
            - computing a RDM
            - a categorization test
            - and an IT regression.

        You will need to EDIT this function to fully complete the assignment.
        Add the necessary analyses as specified in the assignment pdf.
        """
        #retval = {'conv_kernel': results['conv_kernel']}
        retval = {}
        print('Performing neural analysis...')
        meta = self.parse_meta_data(results)
        features, IT_feats = self.get_features(results, num_subsampled_features=1024)

        print('IT:')
        retval['rdm_it'] = \
                self.compute_rdm(IT_feats, meta, mean_objects=True)

        data_subsets = [['V0', 'V3', 'V6'], ['V6']]
        categories = [['Animals'], ['Boats'], ['Cars'], ['Chairs'], 
                      ['Faces'], ['Fruits'], ['Planes'], ['Tables']]
        
        for layer in features:

            print('Layer: %s' % layer)
            data_subset = data_subsets[0]

            retval['rdm_%s' % (layer)] = \
                    self.compute_rdm(features[layer], meta, mean_objects=True)  
 
           # RDM correlation
            retval['spearman_corrcoef_%s' % (layer)] = \
                    spearmanr(
                            np.reshape(retval['rdm_%s' % (layer)], [-1]),
                            np.reshape(retval['rdm_it'], [-1])
                            )[0]
                          
            # IT regression test
            retval['it_regression_%s' % (layer)] = \
                    self.regression_test(features[layer], IT_feats, meta, data_subset)
                
            # categorization test
#            retval['categorization_%s' % (layer)] = \
#                    self.categorization_test(features[layer], meta, data_subset)
                    
            # within-category categorization test
#            for category in categories:
#                retval['within_categorization_%s_%s' % (layer, category[0])] = \
#                        self.within_categorization_test(features[layer], meta, data_subset, category)

            # position regression test
#            retval['pos_regression_%s' % (layer)] = \
#                    self.pos_regression_test(features[layer], meta, data_subset)

        return retval

if __name__ == '__main__':
    """
    Illustrates how to run the configured model using tfutils
    """
    base.get_params()
    m = NeuralDataExperiment()
    params = m.setup_params()
    base.test_from_params(**params)

### general comments on model comparision


Dan suggested we read [Deep convolutional models improve predictions of
macaque V1 responses to natural images](https://www.biorxiv.org/content/biorxiv/early/2017/10/11/201764.full.pdf?%3Fcollection=). they argue that```conv3_1``` explains the most variance in the neural data, given the particular measure they were using (**FEV**): 


> We measured the performance of all models with the fraction of explainable variance explained (FEV). That is, the ratio between the variance accounted for by the model (variance explained) and the explainable variance. 


so i'm setting `out['conv1_kernel'] = out['conv3_1']` after fitting to neural data, this is out benchmark to beat with V1 (if we can get the data). 

. 

. 


#### notes creating `vgg_reference()` in `vgg_retinal_model.py`
 
(1) started with a copy of `vgg16()` 

(2) removed: 

```
with tf.variable_scope('conv0'):
	m.conv(3, 1, 1, in_layer=inputs['images']) # project 40 channels to 3
```

and instead passed inputs to the first conv layer: 

```
with tf.variable_scope('conv1'): 
    out['conv1_1'] = m.conv(64, 3, 1, init=init, init_file=vgg_param_path, in_layer= inputs['images'],     
    init_layer_keys={'weight': 'conv1_1_W', 'bias': 'conv1_1_b'}, trainable=False)
```

(3) changed   `return m.output, {}` to `return out, {}`

(4) removed the fc layer in the retinal model

```
with tf.variable_scope('fc8'):
	out['fc8'] =  m.fc(5, dropout=None, init='trunc_norm', bias=0, 
			stddev=0.05, activation='softplus', weight_decay=1e-3, in_layer=out[target])
```

and just passed passed the inputs along to the next conv layer

(5) uncommented everything at `conv8` and below, passed `conv` and `pool` layers to `out`

(6) set all layers to `trainable=False`


. 

. 

#### notes on `train_vgg_reference.py` 

this script is basically to get this model into the mongo database--basically what we had to do with the gabor model. we're getting this done, basically, by having the script just do one step and then save everything. which is done here:  

	 62             'num_steps': 1, # tyler changed self.Config.train_steps,
	 63             'thres_loss': self.Config.thres_loss,
	 64             'validate_first': True
	 
	 ...  
	 
	 201         params['save_params'] = {
	 202             'host': 'localhost',
	 203             'port': 24444,
	 204             'dbname': database_name,
	 205             'collname': collection_name,
	 206             'exp_id': experiment_ID,
	 207             'save_valid_freq': 1,
	 208             'save_filters_freq': 1,
	 209             'cache_filters_freq': 1,
	 210             'save_metrics_freq': 1,
	 211             'save_initial_filters' : True,



we're using a crop size of 224, by the way ... which is different that alexnet's 227


(1) script is a copy of `/home/biota/gabor_model/train_gabor_model.py` 

(2) set experiment tags: 

		database_name = 'modern_prometheus'
		collection_name = 'vgg_models'
		experiment_ID = 'vgg_reference'
		vgg_function = ' vgg16_reference'




#### notes creating `test_vgg_reference.py` 




(1) script is a copy of `/home/biota/assignment1_code/test_imagenet.py` from assignment one

(2) ported over all layers and set 

		database_name = 'modern_prometheus'
		collection_name = 'vgg_models'
		experiment_ID = 'vgg_reference'
		vgg_function = ' vgg16_reference'

(3) removed `+ 'early'` from line 142 because we're just doing one time step 


**a few questions for Eli:**

doesn't seem like we have access to any of eli's work: 

	ls /home/eliwang/models/VGG16/vgg16_weights.npz
	ls: cannot access '/home/eliwang/models/VGG16/vgg16_weights.npz': No such file or directory 

1. your ```fc8``` isn’t going to be in the final model, right? we’ll chop that off when we put you’re retinal model into the bigger model--but you need it to train the lower layers. 
2. why is ```in_layer=out[target]``` in ```fc8```?  is that the target retinal response? 
3. in your vgg16 model, should ``` return m.output, {}``` should have been ```return out, {}```? 
4. 



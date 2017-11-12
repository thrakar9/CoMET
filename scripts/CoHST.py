# coding: utf-8

# In[1]:

import os
import sys

import numpy as np

seed = 7
np.random.seed(seed)

from keras.utils.np_utils import to_categorical

sys.path.insert(0, os.path.abspath('../Evolutron'))
from evolutron.tools import load_dataset, none2str, Handle, shape, get_args
from evolutron.templates import callback_templates as cb

import nets

# In[7]:

x_pos, _ = load_dataset(data_id='file', padded='x', infile='/data/datasets/uniprot_cas9.tsv')
x_rand, _ = load_dataset(data_id='random', padded=True, max_aa=x_pos.shape[1], min_aa=x_pos.shape[1])
indices = np.random.choice(len(x_rand), size=len(x_pos))
x_neg = x_rand[indices]

# In[51]:

x_train = np.concatenate([x_pos, x_neg], axis=0)
y_train = np.hstack([np.zeros(len(x_pos)), np.ones(len(x_neg))])
y_train = to_categorical(y_train)

# In[52]:

rng_state = np.random.get_state()
np.random.shuffle(x_train)
np.random.set_state(rng_state)
np.random.shuffle(y_train)

# In[55]:

input_shape = x_train[0].shape
output_dim = y_train.shape[1]

conv = 1
fc = 2
filters = [100]
filter_length = [25]
optimizer = 'nadam'
rate = 0.002
batch_size = 256

# In[56]:

conv_net = nets.build_cofam_model(input_shape,
                                  output_dim,
                                  n_conv_layers=conv,
                                  n_fc_layers=fc,
                                  filters=filters,
                                  filter_length=filter_length,
                                  optimizer=optimizer,
                                  lr=rate)

# In[60]:

# Train 100 epochs with validation

epochs = 100
callbacks = cb.standard(patience=20, reduce_factor=.5)
conv_net.fit(x_train, y_train,
             epochs=epochs,
             batch_size=batch_size,
             validation_split=.2,
             callbacks=callbacks,
             return_best_model=True)

# In[ ]:

handle = Handle(conv=conv, fc=fc, filter_length=filter_length, filters=filters, data_id='uniprot_cas9', model='CoHST')
conv_net.save_train_history(handle)
conv_net.save(handle)

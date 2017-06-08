
# coding: utf-8

# In[2]:

import os
import sys
import time

import numpy as np
seed = 7
np.random.seed(seed)

from keras.utils.np_utils import to_categorical
import keras.backend as K

sys.path.insert(0, os.path.abspath('../Evolutron'))
from evolutron.motifs import motif_extraction
from evolutron.tools import load_dataset, none2str, Handle, shape, get_args
from evolutron.templates import callback_templates as cb


import nets

import argparse


# In[7]:

x_train, y_train = load_dataset(data_id='file', padded='True', codes=True, code_key='family',
                                infile='/data/uniprot_fam_g100_s100_train.csv')
x_test, y_test = load_dataset(data_id='file', padded='True', codes=True, code_key='family', max_aa=len(x_train[0]),
                              infile='/data/uniprot_fam_g100_s100_test.csv')
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[8]:

rng_state = np.random.get_state()
np.random.shuffle(x_train)
np.random.set_state(rng_state)
np.random.shuffle(y_train)


# In[9]:

input_shape = x_train[0].shape
output_dim = y_train.shape[1]

parser = argparse.ArgumentParser()
parser.add_argument('--conv', type=int, default=1)
parser.add_argument('--fc', type=int, default=1)
parser.add_argument('--length', type=list, default=[25])
parser.add_argument('--filters', type=list, default=[100])
args = parser.parse_args()

conv=args.conv
fc=args.fc
filters=args.filt
filter_length=args.leng
optimizer = 'nadam'
rate=0.002
batch_size=256


# In[10]:

conv_net = nets.build_cofam_model(input_shape,
                                  output_dim,
                                  n_conv_layers=conv,
                                  n_fc_layers=fc,
                                  filters=filters,
                                  filter_length=filter_length,
                                  optimizer=optimizer,
                                  lr=rate)


# In[11]:

# Train 100 epochs with validation
epochs=100
callbacks = cb.standard(patience=20, reduce_factor=.5)
conv_net.fit(x_train, y_train,
             epochs=epochs,
             batch_size=batch_size,
             validation_data=(x_test, y_test),
             callbacks=callbacks,
             return_best_model=True)


# In[14]:

handle = Handle(conv=conv, fc=fc, filter_length=filter_length,filters=filters,data_id='uniprot_fam_g100_s100',model='DeepCoFAM')
conv_net.save_train_history(handle)
conv_net.save(handle)


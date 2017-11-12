# coding: utf-8

# In[2]:

import os
import sys
import argparse

import numpy as np

seed = 7
np.random.seed(seed)

from keras.utils.np_utils import to_categorical

sys.path.insert(0, os.path.abspath('../Evolutron'))
from evolutron.tools import load_dataset, none2str, Handle, shape, get_args
from evolutron.templates import callback_templates as cb

import nets

parser = argparse.ArgumentParser()
parser.add_argument('--conv', type=int, default=1)
parser.add_argument('--fc', type=int, default=1)
parser.add_argument('--length', type=int, nargs='+')
parser.add_argument('--filters', type=int, nargs='+')
args = parser.parse_args()

conv = args.conv
fc = args.fc
filters = args.filters
filter_length = args.length

x_train, y_train = load_dataset(data_id='file', padded='True', codes=True, code_key='family',
                                infile='/data/uniprot_fam_g100_s100_train.csv')
x_test, y_test = load_dataset(data_id='file', padded='True', codes=True, code_key='family', max_aa=len(x_train[0]),
                              infile='/data/uniprot_fam_g100_s100_test.csv')
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

rng_state = np.random.get_state()
np.random.shuffle(x_train)
np.random.set_state(rng_state)
np.random.shuffle(y_train)

input_shape = x_train[0].shape
output_dim = y_train.shape[1]

optimizer = 'nadam'
rate = 0.002
batch_size = 256

conv_net = nets.build_cofam_model(input_shape,
                                  output_dim,
                                  n_conv_layers=conv,
                                  n_fc_layers=fc,
                                  filters=filters,
                                  filter_length=filter_length,
                                  optimizer=optimizer,
                                  lr=rate)

# Train 100 epochs with validation
epochs = 100
callbacks = cb.standard(patience=20, reduce_factor=.5)
conv_net.fit(x_train, y_train,
             epochs=epochs,
             batch_size=batch_size,
             validation_data=(x_test, y_test),
             callbacks=callbacks,
             return_best_model=True)


handle = Handle(conv=conv, fc=fc, filter_length=filter_length, filters=filters, data_id='uniprot_fam_g100_s100',
                model='DeepCoFAM')
conv_net.save_train_history(handle)
conv_net.save(handle)

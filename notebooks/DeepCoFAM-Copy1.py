
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

sys.path.insert(0, os.path.abspath('../../Evolutron'))
from evolutron.motifs import motif_extraction
from evolutron.tools import load_dataset, none2str, Handle, shape, get_args
from evolutron.engine import DeepTrainer

os.chdir('..')
print(os.listdir('.'))
import nets


# In[3]:

x_train, y_train = load_dataset(data_id='file', padded='True', codes=True, code_key='family',
                                infile='/data/uniprot_fam_g100_s100_train.csv')
x_test, y_test = load_dataset(data_id='file', padded='True', codes=True, code_key='family', max_aa=len(x_train[0]),
                              infile='/data/uniprot_fam_g100_s100_test.csv')
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[5]:

rng_state = np.random.get_state()
np.random.shuffle(x_train)
np.random.set_state(rng_state)
np.random.shuffle(y_train)


# In[9]:

input_shape = x_train[0].shape
output_dim = y_train.shape[1]

conv=1
fc=1
filters=500
filter_length=10
optimizer = 'nadam'
rate=0.002
batch_size=100


# In[10]:

net_arch = nets.DeepCoFAM_test.from_options(input_shape,
                                            output_dim,
                                            n_conv_layers=conv,
                                            n_fc_layers=fc,
                                            n_filters=filters,
                                            filter_length=filter_length)


# In[11]:

conv_net = DeepTrainer(net_arch, classification=True)


# In[12]:

conv_net.compile(optimizer=optimizer, lr=rate)


# In[13]:

conv_net.display_network_info()


# In[14]:

epochs=1
conv_net.fit(x_train, y_train,
             epochs=epochs,
             batch_size=batch_size,
             validation_data=(x_test, y_test),
             patience=20)


# In[ ]:




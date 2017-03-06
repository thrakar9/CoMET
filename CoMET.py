#!/usr/bin/env python
# coding=utf-8
"""
    CoMET - Convolutional Motif Extraction Tool
    ------------------------------------------_
    CoMET is an automated tool for the discovery of protein motifs from arbitrarily
    large protein sequence datasets.

    (c) Massachusetts Institute of Technology

    For more information contact:
    karydis [at] mit.edu
"""
from __future__ import print_function, division, absolute_import

import argparse
import os
import sys
import time

import numpy as np
from keras.utils.np_utils import to_categorical

# Check if package is installed, else fallback to developer mode imports
try:
    from evolutron.motifs import motif_extraction
    from evolutron.tools import load_dataset, none2str, Handle, get_args
    from evolutron.engine import DeepTrainer
except ImportError:

    sys.path.insert(0, os.path.abspath('..'))
    from evolutron.motifs import motif_extraction
    from evolutron.tools import load_dataset, none2str, Handle, shape, get_args
    from evolutron.engine import DeepTrainer

import nets

seed = 7
np.random.seed(seed)


def family(dataset, handle, epochs=1, batch_size=1, filters=30, filter_length=10, validation=.2,
           optimizer='nadam', rate=.01, conv=1, fc=1, model=None):
    # Find input shape
    x_data, y_data = dataset
    if type(x_data) == np.ndarray:
        input_shape = x_data[0].shape
    elif type(x_data) == list:
        input_shape = (None, x_data[0].shape[1])
    else:
        raise TypeError('Something went wrong with the dataset type')

    y_data = to_categorical(y_data)

    output_dim = y_data.shape[1]

    if model:
        conv_net = DeepTrainer(nets.DeepCoFAM.from_saved_model(model))
        print('Loaded model')
    else:
        print('Building model ...')
        net_arch = nets.DeepCoFAM.from_options(input_shape,
                                               output_dim,
                                               n_conv_layers=conv,
                                               n_fc_layers=fc,
                                               n_filters=filters,
                                               filter_length=filter_length)
        handle.model = net_arch.name
        conv_net = DeepTrainer(net_arch, classification=True)
        conv_net.compile(optimizer=optimizer, lr=rate)

    conv_net.display_network_info()

    print('Started training at {}'.format(time.asctime()))

    conv_net.fit(x_data, y_data,
                 nb_epoch=epochs,
                 batch_size=batch_size,
                 validate=validation,
                 patience=50)

    conv_net.save_train_history(handle)
    conv_net.save_model_to_file(handle)

    # Extract the motifs from the convolutional layers
    # motif_extraction(conv_net.custom_fun(), x_data, handle)


def unsupervised(x_data, handle, epochs=1, batch_size=1, filters=30, filter_length=10, validation=.2,
                 optimizer='nadam', rate=.01, conv=1, fc=1, model=None):
    # Find input shape
    if type(x_data) == np.ndarray:
        input_shape = x_data[0].shape
    elif type(x_data) == list:
        input_shape = (None, x_data[0].shape[1])
    else:
        raise TypeError('Something went wrong with the dataset type')

    if model:
        conv_net = DeepTrainer(nets.DeepCoDER.from_saved_model(model))
        print('Loaded model')
    else:
        print('Building model ...')
        net_arch = nets.DeepCoDER.from_options(input_shape,
                                               n_conv_layers=conv,
                                               n_fc_layers=fc,
                                               n_filters=filters,
                                               filter_length=filter_length)
        handle.model = net_arch.name
        conv_net = DeepTrainer(net_arch)
        conv_net.compile(optimizer=optimizer, lr=rate)

    conv_net.display_network_info()

    print('Started training at {}'.format(time.asctime()))

    conv_net.fit(x_data, x_data,
                 nb_epoch=epochs,
                 batch_size=batch_size,
                 validate=validation,
                 patience=20)

    conv_net.save_train_history(handle)
    conv_net.save_model_to_file(handle)

    # Extract the motifs from the convolutional layers
    # motif_extraction(conv_net.custom_fun(), x_data, handle)


def main(mode, **options):
    if 'model' in options:
        handle = Handle.from_filename(options.get('model'))
        assert handle.program == 'CoMET', 'The model file provided is for another program.'
    else:
        handle = Handle(**options)

    # Load the dataset
    print("Loading data...")
    dataset_options = get_args(options, ['data_id', 'padded'])

    if mode == 'unsupervised':
        dataset = load_dataset(**dataset_options)
        unsupervised(dataset, handle, **options)
    elif mode == 'family':
        dataset = load_dataset(**dataset_options, codes=True)
        family(dataset, handle, **options)
    else:
        raise IOError('Invalid mode of operation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CoMET - Convolutional Motif Embeddings Tool',
                                     argument_default=argparse.SUPPRESS)

    parser.add_argument("data_id",
                        help='The protein dataset to be trained on.')

    parser.add_argument("filters", type=int,
                        help='Number of filters in the convolutional layers.')

    parser.add_argument("filter_length", type=int,
                        help='Size of filters in the first convolutional layer.')

    parser.add_argument("--no_pad", action='store_true',
                        help='Toggle to pad protein sequences. Batch size auto-change to 1.')

    parser.add_argument("--mode", default='unsupervised')

    parser.add_argument("--conv", type=int, default=1,
                        help='number of conv layers.')

    parser.add_argument("--fc", type=int, default=1,
                        help='number of fc layers.')

    parser.add_argument("-e", "--epochs", default=50, type=int,
                        help='number of training epochs to perform (default: 50)')

    parser.add_argument("-b", "--batch_size", type=int, default=50,
                        help='Size of minibatch.')

    parser.add_argument("--rate", type=float,
                        help='The learning rate for the optimizer.')

    parser.add_argument("--model", type=str,
                        help='Continue training the given model. Other architecture options are unused.')

    parser.add_argument("--optimizer", choices=['adam', 'nadam', 'rmsprop', 'sgd', 'adadelta', 'adagrad'],
                        help='The optimizer to be used.')

    args = parser.parse_args()

    kwargs = args.__dict__

    if hasattr(args, 'no_pad'):
        kwargs['batch_size'] = 1
        kwargs.pop('no_pad')
        kwargs['padded'] = False

    if hasattr(args, 'model'):
        kwargs.pop('filters')
        kwargs.pop('filter_length')

    main(**kwargs)

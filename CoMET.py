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
import argparse
import os
import sys
import time

import numpy as np
from keras.utils.np_utils import to_categorical
import keras.backend as K

# Check if package is installed, else fallback to developer mode imports
try:
    from evolutron.motifs import motif_extraction
    from evolutron.tools import load_dataset, none2str, Handle, shape, get_args
except ImportError:
    sys.path.insert(0, os.path.abspath('../Evolutron'))
    from evolutron.motifs import motif_extraction
    from evolutron.tools import load_dataset, none2str, Handle, shape, get_args
    from evolutron.templates import callback_templates as cb

import nets

seed = 7
np.random.seed(seed)


def family(dataset, handle, epochs=1, batch_size=1, filters=30, filter_length=10, validation=.2,
           optimizer='nadam', rate=.01, conv=1, fc=1, model=None, motifs=True):
    # TODO: be able to submit train and test files separately
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
        # conv_net = DeepTrainer(nets.DeepCoFAM.from_saved_model(model))
        conv_net = None
        # TODO: implement load model
        print('Loaded model')
    else:
        print('Building model ...')
        conv_net = nets.build_cofam_model(input_shape,
                                          output_dim,
                                          n_conv_layers=conv,
                                          n_fc_layers=fc,
                                          filters=filters,
                                          filter_length=filter_length,
                                          optimizer=optimizer,
                                          lr=rate)

    handle.model = conv_net.name

    conv_net.display_network_info()

    callbacks = cb.standard(patience=20, reduce_factor=.05)

    print('Started training at {}'.format(time.asctime()))
    conv_net.fit(x_data, y_data,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_split=validation,
                 callbacks=callbacks)

    conv_net.save_train_history(handle)
    conv_net.save_model_to_file(handle)

    # Extract the motifs from the convolutional layers
    if motifs:
        # TODO: hide custom fun in deep trainer
        for depth, conv_layer in enumerate(conv_net.get_conv_layers()):
            conv_scores = conv_layer.output
            # Compile function that spits out the outputs of the correct convolutional layer
            boolean_mask = K.any(K.not_equal(conv_net.input, 0.0), axis=-1, keepdims=True)
            conv_scores = conv_scores * K.cast(boolean_mask, K.floatx())

            custom_fun = K.function([conv_net.input], [conv_scores])
            # Start visualizations
            motif_extraction(custom_fun, x_data, conv_layer.filters,
                             conv_layer.kernel_size[0], handle, depth)


def unsupervised(dataset, handle, epochs=1, batch_size=1, filters=30, filter_length=10, validation=.2,
                 optimizer='nadam', rate=.005, conv=1, fc=1, model=None, motifs=True):
    x_data = dataset[0]
    # Find input shape
    if type(x_data) == np.ndarray:
        input_shape = x_data[0].shape
    elif type(x_data) == list:
        input_shape = (None, x_data[0].shape[1])
    else:
        raise TypeError('Something went wrong with the dataset type')

    if model:
        conv_net = None
        # TODO: implement load model
        print('Loaded model')
    else:
        print('Building model ...')
        conv_net = nets.build_coder_model(input_shape,
                                          n_conv_layers=conv,
                                          n_fc_layers=fc,
                                          filters=filters,
                                          filter_length=filter_length,
                                          optimizer=optimizer,
                                          lr=rate)
    handle.model = conv_net.name
    conv_net.display_network_info()

    callbacks = cb.standard(patience=20, reduce_factor=.05)

    print('Started training at {}'.format(time.asctime()))

    conv_net.fit(x_data, x_data,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_split=validation,
                 callbacks=callbacks)

    conv_net.save_train_history(handle)
    conv_net.save_model_to_file(handle)

    # Extract the motifs from the convolutional layers
    if motifs:
        # TODO: hide custom fun in deep trainer
        for depth, conv_layer in enumerate(conv_net.get_conv_layers()):
            conv_scores = conv_layer.output
            # Compile function that spits out the outputs of the correct convolutional layer
            boolean_mask = K.any(K.not_equal(conv_net.input, 0.0), axis=-1, keepdims=True)
            conv_scores = conv_scores * K.cast(boolean_mask, K.floatx())

            custom_fun = K.function([conv_net.input], [conv_scores])
            # Start visualizations
            motif_extraction(custom_fun, x_data, conv_layer.filters,
                             conv_layer.kernel_size[0], handle, depth)


def main(mode, **options):
    if 'model' in options:
        handle = Handle.from_filename(options.get('model'))
        assert handle.ftype == 'model'
        assert handle.model in ['DeepCoDER', 'DeepCoFAM'], 'The model file provided is for another program.'
    else:
        handle = Handle(**options)

    # Load the dataset
    print("Loading data...")
    dataset_options = get_args(options, ['data_id', 'padded', 'infile'])

    if mode == 'unsupervised' or handle.model == 'DeepCoDER':
        dataset = load_dataset(**dataset_options)
        unsupervised(dataset, handle, **options)
    elif mode == 'family' or handle.model == 'DeepCoFAM':
        dataset = load_dataset(**dataset_options, codes=True, code_key=options.pop('key', 'fam'))
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

    parser.add_argument("--infile", "-i",
                        help='The protein dataset file to be trained on. Only if data_id = file')

    parser.add_argument("--key",
                        help='The key to use for codes.')

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

    if args.data_id == 'file' and not hasattr(args, 'infile'):
        raise argparse.ArgumentError('Data id is set to "file", but no --path options. Call with -h for help.')

    main(**kwargs)

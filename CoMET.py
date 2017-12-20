#!/usr/bin/env python
# coding=utf-8
"""
    CoMET - Convolutional Motif Extraction Tool
    -------------------------------------------
    CoMET is an automated tool for the discovery of protein motifs from arbitrarily
    large protein sequence datasets.

    (c) 2015-2017 Massachusetts Institute of Technology

    For more information contact:
    karydis [at] mit.edu
"""
import os
import sys
import time

import dill
import keras.backend as K
import numpy as np
import pandas as pd
from absl import flags
from keras.utils.np_utils import to_categorical

import nets
from evolutron.motifs import motif_extraction
from evolutron.templates import callback_templates as cb
from evolutron.tools import Handle, load_dataset, load_random_aa_seqs, preprocess_dataset

flags.DEFINE_string("infile", '', 'The protein dataset file to be trained on.')
flags.DEFINE_string("cohst_neg_file", '', 'The protein dataset file to use as a negative set on CoHST')

flags.DEFINE_string("key", 'fam', 'The key to use for codes.')
flags.DEFINE_boolean("no_pad", False, 'Toggle to disable padding protein sequences. Batch size will auto-change to 1.')
flags.DEFINE_integer("pad_length", None, 'The max length to use for proteins in the dataset.')
flags.DEFINE_integer("dataset_size", None, 'The number of samples to use. If None, dataset_fraction=1.0 will be used.')
flags.DEFINE_float("dataset_fraction", 1.0,
                   'The fraction of the dataset to use. Option is overwritten if dataset_size is set.')

flags.DEFINE_enum("mode", 'CoDER', ['CoDER', 'CoFAM', 'CoHST'], 'The mode to train CoMET.')
flags.DEFINE_integer("epochs", 50, 'The number of training epochs to perform.', lower_bound=1)
flags.DEFINE_integer("batch_size", 50, 'The size of the mini-batch.', lower_bound=1)
flags.DEFINE_float("validation_split", 0.2, "The fraction of data to use for cross-validation.", lower_bound=0.0,
                   upper_bound=1.0)

flags.DEFINE_string("model", '', 'Continue training the given model. Other architecture options are unused.')

flags.DEFINE_boolean("motifs", True, 'Toggle to enable/disable motif extraction.')
flags.DEFINE_enum("motifs_filetype", 'txt+png', ['png', 'pdf', 'txt', 'txt+pdf', 'txt+png'],
                  'Choose between different file types to save the extracted motifs from CoMET.'
                  'A typical workflow for subsequent analysis would be to extract the motifs as text files (txt) and'
                  'then use the tool sites2meme to transform them to MEME format and submit for search in MAST.')

flags.DEFINE_string("data_dir", '', 'The directory to store CoMET output data.')

FLAGS = flags.FLAGS

try:
    FLAGS(sys.argv)
except flags.Error as e:
    print(e)
    print(FLAGS)
    sys.exit(1)


def extract_motifs(x_data, conv_net, handle):
    for depth, conv_layer in enumerate(conv_net.get_conv_layers()):
        conv_scores = conv_layer.output
        # Compile function that spits out the outputs of the correct convolutional layer
        boolean_mask = K.any(K.not_equal(conv_net.input, 0.0), axis=-1, keepdims=True)
        conv_scores = conv_scores * K.cast(boolean_mask, K.floatx())

        custom_fun = K.function([conv_net.input], [conv_scores])
        # Start visualizations
        motif_extraction(custom_fun, x_data, conv_layer.filters, conv_layer.kernel_size[0], handle, depth,
                         filetype=FLAGS.motifs_filetype)


def save_experiment(conv_net):
    file_key = str(np.random.randint(10 ** 9, 10 ** 10))

    conv_net.save_train_history(file_key, data_dir=FLAGS.data_dir)
    conv_net.save(file_key, data_dir=FLAGS.data_dir)
    conv_net.save_architecture(file_key, data_dir=FLAGS.data_dir)
    dill.dump(FLAGS.flag_values_dict(),
              open(os.path.join(FLAGS.data_dir, 'models', file_key + '.flags'), 'wb'))
    return file_key


def binary(x_data, y_data, handle):
    # Find input shape
    if type(x_data) == np.ndarray:
        input_shape = x_data[0].shape
        FLAGS.pad_length = x_data[0].shape[0]
    elif type(x_data) == list:
        input_shape = (None, x_data[0].shape[1])
        FLAGS.pad_length = -1
    else:
        raise TypeError('Something went wrong with the dataset type')

    if FLAGS.model:
        conv_net = nets.build_cohst_model(saved_model=FLAGS.model)
        print('Loaded model')
    else:
        print('Building model ...')
        conv_net = nets.build_cohst_model(input_shape)

    handle.model = conv_net.name
    conv_net.display_network_info()

    callbacks = cb.standard(patience=FLAGS.patience, reduce_factor=FLAGS.reduce_factor)

    print('Started training at {}'.format(time.asctime()))
    conv_net.fit(x_data, y_data,
                 epochs=FLAGS.epochs,
                 batch_size=FLAGS.batch_size,
                 validation_split=FLAGS.validation_split,
                 callbacks=callbacks)

    model_key = save_experiment(conv_net)

    # Extract the motifs from the convolutional layers
    if FLAGS.motifs:
        dataset_key = FLAGS.infile.split('/')[-1].split('.')[0]
        output_folder = os.path.join(FLAGS.data_dir, 'motifs', dataset_key, model_key)
        extract_motifs(x_data, conv_net, output_folder)


def family(x_data, y_data, handle):
    # TODO: be able to submit train and test files separately
    # Find input shape
    if type(x_data) == np.ndarray:
        input_shape = x_data[0].shape
        FLAGS.pad_length = x_data[0].shape[0]
    elif type(x_data) == list:
        input_shape = (None, x_data[0].shape[1])
        FLAGS.pad_length = -1
    else:
        raise TypeError('Something went wrong with the dataset type')

    y_data = to_categorical(y_data)

    output_dim = y_data.shape[1]

    if FLAGS.model:
        conv_net = nets.build_cofam_model(saved_model=FLAGS.model)
        print('Loaded model')
    else:
        print('Building model ...')
        conv_net = nets.build_cofam_model(input_shape,
                                          output_dim)

    handle.model = conv_net.name
    conv_net.display_network_info()

    callbacks = cb.standard(patience=FLAGS.patience, reduce_factor=FLAGS.reduce_factor)

    print('Started training at {}'.format(time.asctime()))
    conv_net.fit(x_data, y_data,
                 epochs=FLAGS.epochs,
                 batch_size=FLAGS.batch_size,
                 validation_split=FLAGS.validation_split,
                 callbacks=callbacks)

    model_key = save_experiment(conv_net)

    # Extract the motifs from the convolutional layers
    if FLAGS.motifs:
        dataset_key = FLAGS.infile.split('/')[-1].split('.')[0]
        output_folder = os.path.join(FLAGS.data_dir, 'motifs', dataset_key, model_key)
        extract_motifs(x_data, conv_net, output_folder)


def unsupervised(x_data, handle):
    # Find input shape
    if type(x_data) == np.ndarray:
        input_shape = x_data[0].shape
        FLAGS.pad_length = x_data[0].shape[0]
    elif type(x_data) == list:
        input_shape = (None, x_data[0].shape[1])
        FLAGS.pad_length = -1
    else:
        raise TypeError('Something went wrong with the dataset type')

    if FLAGS.model:
        conv_net = nets.build_coder_model(saved_model=FLAGS.model)
        print('Loaded model')
    else:
        print('Building model ...')
        conv_net = nets.build_coder_model(input_shape)

    handle.model = conv_net.name
    conv_net.display_network_info()

    callbacks = cb.standard(patience=FLAGS.patience, reduce_factor=FLAGS.reduce_factor)

    print('Started training at {}'.format(time.asctime()))

    conv_net.fit(x_data, x_data,
                 epochs=FLAGS.epochs,
                 batch_size=FLAGS.batch_size,
                 validation_split=FLAGS.validation_split,
                 callbacks=callbacks)

    model_key = save_experiment(conv_net)

    # Extract the motifs from the convolutional layers
    if FLAGS.motifs:
        dataset_key = FLAGS.infile.split('/')[-1].split('.')[0]
        output_folder = os.path.join(FLAGS.data_dir, 'motifs', dataset_key, model_key)
        extract_motifs(x_data, conv_net, output_folder)


def main():
    if FLAGS.no_pad:
        FLAGS.batch_size = 1

    if FLAGS.model:
        handle = Handle.from_filename(FLAGS.model)
        assert handle.ftype == 'model'
        assert handle.model in ['DeepCoDER', 'DeepCoFAM', 'CoHST'], 'The model file provided is for another program.'
    else:
        handle = Handle(**FLAGS.flag_values_dict())

    # Load the dataset
    print("Loading data...")

    if FLAGS.mode == 'CoDER' or handle.model == 'CoDER':
        x_data, _ = load_dataset(FLAGS.infile)
        x_data = preprocess_dataset(x_data, padded=not FLAGS.no_pad)
        FLAGS.dataset_size = len(x_data)
        unsupervised(x_data, handle)
    elif FLAGS.mode == 'CoFAM' or handle.model == 'CoFAM':
        x_data, y_data = load_dataset(FLAGS.infile, codes=True, code_key=FLAGS.key)
        x_data, y_data = preprocess_dataset(x_data, y_data, padded=not FLAGS.no_pad)
        FLAGS.dataset_size = len(x_data)
        family(x_data, y_data, handle)
    elif FLAGS.mode == 'CoHST' or handle.model == 'CoHST':
        x_pos, _ = load_dataset(FLAGS.infile)
        if FLAGS.cohst_neg_file:
            x_neg, _ = load_dataset(FLAGS.cohst_neg_file)
        else:
            x_neg = load_random_aa_seqs(len(x_pos), x_pos.str.len().min(), x_pos.str.len().max())
        x_data = pd.concat((x_pos, x_neg), ignore_index=True)
        y_data = [1] * len(x_pos) + [0] * len(x_neg)
        x_data, y_data = preprocess_dataset(x_data, y_data, padded=not FLAGS.no_pad)
        FLAGS.dataset_size = len(x_data)
        binary(x_data, y_data, handle)
    else:
        raise IOError('Invalid mode of operation.')


if __name__ == '__main__':
    main()

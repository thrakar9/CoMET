"""
    Implementation of CoMET in Keras.

    Thrasyvoulos Karydis
    (c) Massachusetts Institute of Technology 2016-2017

    This work may be reproduced, modified, distributed, performed, and
    displayed for any purpose, but must acknowledge the mods
    project. Copyright is retained and must be preserved. The work is
    provided as is; no warranty is provided, and users accept all
    liability.
"""
import keras.backend as K
from absl import flags
from keras import regularizers
from keras.layers import Activation, BatchNormalization, Convolution1D, Dense, Flatten, GlobalMaxPooling1D, Input, \
    MaxPooling1D, Reshape
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.metrics import binary_accuracy, categorical_accuracy

from evolutron.engine import Model, load_model
from evolutron.extra_layers import Deconvolution1D, Dedense, Upsampling1D, custom_layers
from evolutron.extra_metrics import mean_cat_acc
from evolutron.extra_objectives import masked_mse
from . import network_parameters

flags.adopt_module_key_flags(network_parameters)

FLAGS = flags.FLAGS


def build_coder_model(input_shape=None, saved_model=None):
    if saved_model:
        model = load_model(saved_model, custom_objects=custom_layers, compile=False)
    else:
        # Check input parameters
        assert len(input_shape) == 2, 'Unrecognizable input dimensions'
        assert K.image_dim_ordering() == 'tf', 'Theano dimension ordering not supported yet'
        assert input_shape[1] in [20, 4, 22], 'Input dimensions error, check order'

        seq_length, alphabet = input_shape

        # Input LayerRO
        inp = Input(shape=input_shape, name='aa_seq')
        feature_layer = inp
        # Convolutional Layers
        convs = []
        for i, (filter_number, filter_size, filter_stride) in enumerate(
                zip(FLAGS.filters, FLAGS.filter_length, FLAGS.filter_stride)):
            feature_layer = Convolution1D(filters=filter_number,
                                          kernel_size=filter_size,
                                          strides=filter_stride,
                                          padding='same',
                                          use_bias=False,
                                          kernel_initializer='glorot_uniform',
                                          activation='linear',
                                          name='Conv%d' % (i + 1))(feature_layer)
            convs.append(feature_layer)
            feature_layer = BatchNormalization()(feature_layer)
            feature_layer = Activation(activation='relu')(feature_layer)

        # Global Max-pooling
        pool = GlobalMaxPooling1D()(feature_layer)

        # Fully-Connected encoding layers
        fc_enc = [Dense(FLAGS.filters[-1],
                        kernel_initializer='glorot_uniform',
                        activation='sigmoid',
                        name='FCEnc1')(pool)]

        for d in range(1, FLAGS.n_fc_layers):
            fc_enc.append(Dense(FLAGS.filters[-1],
                                kernel_initializer='glorot_uniform',
                                activation='sigmoid',
                                name='FCEnc{}'.format(d + 1))(fc_enc[-1]))

        encoded = fc_enc[-1]  # To access if model for encoding needed

        # Fully-Connected decoding layers
        fc_dec = [Dedense(encoded._keras_history[0],
                          activation='linear',
                          name='FCDec{}'.format(FLAGS.n_fc_layers))(encoded)]

        for d in range(FLAGS.n_fc_layers - 2, -1, -1):
            fc_dec.append(Dedense(fc_enc[d]._keras_history[0],
                                  activation='linear',
                                  name='FCDec{}'.format(d + 1))(fc_dec[-1]))

        # Reshaping and unpooling
        unflat = Reshape((1, fc_dec[-1]._keras_shape[-1]))(fc_dec[-1])
        deconvs = [Upsampling1D(pool._keras_history[0].input_shape[1], name='Upsampling')(unflat)]
        # Deconvolution
        for c in range(FLAGS.n_conv_layers - 1, 0, -1):
            deconvs.append(Deconvolution1D(convs[c]._keras_history[0],
                                           activation='relu',
                                           name='Deconv{}'.format(c + 1))(deconvs[-1]))  # maybe add L1 regularizer

        decoded = Deconvolution1D(convs[0]._keras_history[0],
                                  apply_mask=False,
                                  activation='sigmoid',
                                  name='Deconv1')(deconvs[-1])

        model = Model(inputs=inp, outputs=decoded, name='CoDER', classification=False)

    losses = [masked_mse]

    # Metrics
    metrics = [mean_cat_acc]

    # Compilation

    model.compile(optimizer=FLAGS.optimizer,
                  loss=losses,
                  metrics=metrics,
                  lr=FLAGS.learning_rate)
    return model


def build_cofam_model(input_shape=None, output_dim=None, saved_model=None):
    if saved_model:
        model = load_model(saved_model, custom_objects=custom_layers, compile=False)
        model.classification = True
    else:
        # Check input parameters
        assert len(input_shape) == 2, 'Unrecognizable input dimensions'
        assert K.image_dim_ordering() == 'tf', 'Theano dimension ordering not supported yet'
        assert input_shape[1] in [20, 4, 22], 'Input dimensions error, check order'

        seq_length, alphabet = input_shape

        # Model Architecture

        # Input LayerRO
        inp = Input(shape=input_shape, name='aa_seq')
        feature_layer = inp
        # Convolutional Layers
        for i, (filter_number, filter_size, filter_stride) in enumerate(
                zip(FLAGS.filters, FLAGS.filter_length, FLAGS.filter_stride)):
            feature_layer = Convolution1D(filters=filter_number,
                                          kernel_size=filter_size,
                                          strides=filter_stride,
                                          padding='same',
                                          use_bias=False,
                                          kernel_initializer='glorot_uniform',
                                          activation='linear',
                                          name='Conv%d' % (i + 1))(feature_layer)
            feature_layer = BatchNormalization()(feature_layer)
            feature_layer = Activation(activation='relu')(feature_layer)
            # feature_layer = Dropout(rate=0.5)(feature_layer)

        # Max-pooling
        if seq_length:
            max_pool = MaxPooling1D(pool_size=seq_length)(feature_layer)
            flat = Flatten()(max_pool)
        else:
            # max_pool = GlobalMaxPooling1D()(convs[-1])
            # flat = max_pool
            raise NotImplementedError('Sequence length must be known at this point. Pad and use mask.')

        # Fully-Connected encoding layers
        fc_enc = [Dense(FLAGS.filters[-1],
                        kernel_initializer='glorot_uniform',
                        activation='relu',
                        name='FCEnc1')(flat)]

        for d in range(1, FLAGS.n_fc_layers):
            fc_enc.append(Dense(FLAGS.filters[-1],
                                kernel_initializer='glorot_uniform',
                                activation='relu',
                                name='FCEnc{}'.format(d + 1))(fc_enc[-1]))

        encoded = fc_enc[-1]  # To access if model for encoding needed

        classifier = Dense(output_dim,
                           kernel_initializer='glorot_uniform',
                           activation='softmax',
                           name='Classifier')(encoded)

        model = Model(inputs=inp, outputs=classifier, name='CoFAM', classification=True)

    # Loss Functions
    losses = [categorical_crossentropy]

    # Metrics
    metrics = [categorical_accuracy]

    # Compilation

    model.compile(optimizer=FLAGS.optimizer,
                  loss=losses,
                  metrics=metrics,
                  lr=FLAGS.learning_rate)
    return model


def build_cohst_model(input_shape=None, saved_model=None):
    if saved_model:
        model = load_model(saved_model, custom_objects=custom_layers, compile=False)
        model.classification = True
    else:
        # Check input parameters
        assert len(input_shape) == 2, 'Unrecognizable input dimensions'
        assert K.image_dim_ordering() == 'tf', 'Theano dimension ordering not supported yet'
        assert input_shape[1] in [20, 4, 22], 'Input dimensions error, check order'

        seq_length, alphabet = input_shape

        # Model Architecture

        # Input LayerRO
        inp = Input(shape=input_shape, name='aa_seq')
        feature_layer = inp
        # Convolutional Layers
        for i, (filter_number, filter_size, filter_stride) in enumerate(
                zip(FLAGS.filters, FLAGS.filter_length, FLAGS.filter_stride)):
            feature_layer = Convolution1D(filters=filter_number,
                                          kernel_size=filter_size,
                                          strides=filter_stride,
                                          padding='same',
                                          use_bias=False,
                                          kernel_initializer='glorot_uniform',
                                          activation='linear',
                                          name='Conv%d' % (i + 1))(feature_layer)
            feature_layer = BatchNormalization()(feature_layer)
            feature_layer = Activation(activation='relu')(feature_layer)

        # Max-pooling
        if seq_length:
            max_pool = MaxPooling1D(pool_size=seq_length)(feature_layer)
            flat = Flatten()(max_pool)
        else:
            # max_pool = GlobalMaxPooling1D()(convs[-1])
            # flat = max_pool
            raise NotImplementedError('Sequence length must be known at this point. Pad and use mask.')

        # Fully-Connected encoding layers
        fc_enc = [Dense(FLAGS.filters[-1],
                        kernel_initializer='glorot_uniform',
                        activation='relu', activity_regularizer=regularizers.l2(0.02),
                        name='FCEnc1')(flat)]

        for d in range(1, FLAGS.n_fc_layers):
            fc_enc.append(Dense(FLAGS.filters[-1],
                                kernel_initializer='glorot_uniform',
                                activation='relu', activity_regularizer=regularizers.l2(0.02),
                                name='FCEnc{}'.format(d + 1))(fc_enc[-1]))

        encoded = fc_enc[-1]  # To access if model for encoding needed

        classifier = Dense(1, activation='sigmoid',
                           name='Classifier')(encoded)

        model = Model(inputs=inp, outputs=classifier, name='CoHST', classification=True)

    # Loss Functions
    losses = [binary_crossentropy]

    # Metrics
    metrics = [binary_accuracy]

    # Compilation

    model.compile(optimizer=FLAGS.optimizer,
                  loss=losses,
                  metrics=metrics,
                  lr=FLAGS.learning_rate)
    return model

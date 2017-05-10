"""
    Implementation of a Convolutional Autoencoder
    in Keras.

    Thrasyvoulos Karydis
    (c) Massachusetts Institute of Technology 2016

    This work may be reproduced, modified, distributed, performed, and
    displayed for any purpose, but must acknowledge the mods
    project. Copyright is retained and must be preserved. The work is
    provided as is; no warranty is provided, and users accept all
    liability.
"""
import keras.backend as K
from keras.layers import Convolution1D, MaxPooling1D, Dense, Flatten, Reshape
from keras.layers import Input
from keras.layers import BatchNormalization
from evolutron.extra_layers import Dedense, Upsampling1D, Deconvolution1D

from keras.metrics import categorical_accuracy
from evolutron.extra_metrics import mean_cat_acc

from keras.losses import mean_squared_error, categorical_crossentropy

from evolutron.engine import Model


def build_coder_model(input_shape, filters, filter_length, n_conv_layers=1, n_fc_layers=1,
                      optimizer='sgd', lr=.001):
    # Check input parameters
    assert len(input_shape) == 2, 'Unrecognizable input dimensions'
    assert K.image_dim_ordering() == 'tf', 'Theano dimension ordering not supported yet'
    assert input_shape[1] in [20, 4, 22], 'Input dimensions error, check order'

    seq_length, alphabet = input_shape

    # Input LayerRO
    inp = Input(shape=input_shape, name='aa_seq')

    # Convolutional Layers
    convs = [Convolution1D(filters=filters,
                           kernel_size=filter_length,
                           kernel_initializer='glorot_uniform',
                           activation='relu',
                           padding='same',
                           name='Conv1')(inp)]

    for c in range(1, n_conv_layers):
        convs.append(Convolution1D(filters=filters,
                                   kernel_size=filter_length,
                                   kernel_initializer='glorot_uniform',
                                   activation='relu',
                                   padding='same',
                                   name='Conv{}'.format(c + 1))(convs[-1]))  # maybe add L1 regularizer

    # Max-pooling
    if seq_length:
        max_pool = MaxPooling1D(pool_size=seq_length)(convs[-1])
        flat = Flatten()(max_pool)
    else:
        # max_pool = GlobalMaxPooling1D()(convs[-1])
        # flat = max_pool
        raise NotImplementedError('Sequence length must be known at this point. Pad and use mask.')

    # Fully-Connected encoding layers
    fc_enc = [Dense(filters,
                    kernel_initializer='glorot_uniform',
                    activation='sigmoid',
                    name='FCEnc1')(flat)]

    for d in range(1, n_fc_layers):
        fc_enc.append(Dense(filters,
                            kernel_initializer='glorot_uniform',
                            activation='sigmoid',
                            name='FCEnc{}'.format(d + 1))(fc_enc[-1]))

    encoded = fc_enc[-1]  # To access if model for encoding needed

    # Fully-Connected decoding layers
    fc_dec = [Dedense(encoded._keras_history[0],
                      activation='linear',
                      name='FCDec{}'.format(n_fc_layers))(encoded)]

    for d in range(n_fc_layers - 2, -1, -1):
        fc_dec.append(Dedense(fc_enc[d]._keras_history[0],
                              activation='linear',
                              name='FCDec{}'.format(d + 1))(fc_dec[-1]))

    # Reshaping and unpooling
    if seq_length:
        unflat = Reshape(max_pool._keras_shape[1:])(fc_dec[-1])
    else:
        unflat = Reshape((1, fc_dec[-1]._keras_shape[-1]))(fc_dec[-1])

    deconvs = [Upsampling1D(max_pool._keras_history[0].input_shape[1], name='Upsampling')(unflat)]

    # Deconvolution
    for c in range(n_conv_layers - 1, 0, -1):
        deconvs.append(Deconvolution1D(convs[c]._keras_history[0],
                                       activation='relu',
                                       name='Deconv{}'.format(c + 1))(deconvs[-1]))  # maybe add L1 regularizer

    decoded = Deconvolution1D(convs[0]._keras_history[0],
                              apply_mask=False,
                              activation='sigmoid',
                              name='Deconv1')(deconvs[-1])

    model = Model(inputs=inp, outputs=decoded, name='DeepCoDER', classification=False)

    # Loss Functions
    def _loss_function(inp, decoded):
        boolean_mask = K.any(K.not_equal(inp, 0.0), axis=-1, keepdims=True)
        decoded = decoded * K.cast(boolean_mask, K.floatx())
        return mean_squared_error(y_true=inp, y_pred=decoded)

    losses = [_loss_function]

    # Metrics
    metrics = [mean_cat_acc]

    # Compilation

    model.compile(optimizer=optimizer,
                  loss=losses,
                  metrics=metrics,
                  lr=lr)

    return model


def build_cofam_model(input_shape, output_dim, filters, filter_length, n_conv_layers=1, n_fc_layers=1,
                      optimizer='sgd', lr=.001):
    # Check input parameters
    assert len(input_shape) == 2, 'Unrecognizable input dimensions'
    assert K.image_dim_ordering() == 'tf', 'Theano dimension ordering not supported yet'
    assert input_shape[1] in [20, 4, 22], 'Input dimensions error, check order'

    seq_length, alphabet = input_shape

    # Model Architecture

    # Input LayerRO
    inp = Input(shape=input_shape, name='aa_seq')

    # Convolutional Layers
    # First
    convs = [BatchNormalization()(Convolution1D(filters=filters,
                                                kernel_size=filter_length,
                                                strides=1,
                                                kernel_initializer='glorot_uniform',
                                                activation='relu',
                                                use_bias=False,
                                                padding='same',
                                                name='Conv1')(inp))]
    # Middle
    for c in range(1, n_conv_layers):
        convs.append(Convolution1D(filters=filters,
                                   kernel_size=filter_length,
                                   strides=1,
                                   kernel_initializer='glorot_uniform',
                                   activation='relu',
                                   use_bias=False,
                                   padding='same',
                                   name='Conv{}'.format(c + 1))(convs[-1]))  # maybe add L1 regularizer

    # Max-pooling
    if seq_length:
        max_pool = MaxPooling1D(pool_size=seq_length)(convs[-1])
        flat = Flatten()(max_pool)
    else:
        # max_pool = GlobalMaxPooling1D()(convs[-1])
        # flat = max_pool
        raise NotImplementedError('Sequence length must be known at this point. Pad and use mask.')

    # Fully-Connected encoding layers
    fc_enc = [Dense(filters,
                    kernel_initializer='glorot_uniform',
                    activation='relu',
                    name='FCEnc1')(flat)]

    for d in range(1, n_fc_layers):
        fc_enc.append(Dense(filters,
                            kernel_initializer='glorot_uniform',
                            activation='relu',
                            name='FCEnc{}'.format(d + 1))(fc_enc[-1]))

    encoded = fc_enc[-1]  # To access if model for encoding needed

    classifier = Dense(output_dim,
                       kernel_initializer='glorot_uniform',
                       activation='softmax',
                       name='Classifier')(encoded)

    model = Model(inputs=inp, outputs=classifier, name='DeepCoFAM', classification=True)

    # Loss Functions
    losses = [categorical_crossentropy]

    # Metrics
    metrics = [categorical_accuracy]

    # Compilation

    model.compile(optimizer=optimizer,
                  loss=losses,
                  metrics=metrics,
                  lr=lr)

    return model

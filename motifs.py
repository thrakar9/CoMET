#!/usr/bin/env python

import argparse
import h5py
import os
import sys

from keras.models import model_from_json
import keras.backend as K

# Check if package is installed, else fallback to developer mode imports
try:
    from evolutron.engine import DeepTrainer
    from evolutron.tools import load_dataset, Handle, shape
    from evolutron.motifs import motif_extraction
    from evolutron.templates import custom_layers
except ImportError:
    sys.path.insert(0, os.path.abspath('../Evolutron'))
    from evolutron.engine import DeepTrainer
    from evolutron.tools import load_dataset, Handle, shape
    from evolutron.motifs import motif_extraction
    from evolutron.templates import custom_layers


def main(filename, data_id):
    # First load model architecture
    hf = h5py.File(filename)
    model_config = hf.attrs['model_config'].decode('utf8')
    hf.close()

    net = DeepTrainer(model_from_json(model_config, custom_objects=custom_layers))

    # Then load model parameters
    net.load_all_param_values(filename)

    handle = Handle.from_filename(filename)

    if data_id == 'model':
        data_id = handle.dataset

    x_data, y_data = load_dataset(data_id, padded=True, max_aa=net.input._keras_shape[1])

    conv_layers = net.get_conv_layers()

    for depth, conv_layer in enumerate(conv_layers):
        conv_scores = conv_layer.output
        # Compile function that spits out the outputs of the correct convolutional layer
        boolean_mask = K.any(K.not_equal(net.input, 0.0), axis=-1, keepdims=True)
        conv_scores = conv_scores * K.cast(boolean_mask, K.floatx())

        custom_fun = K.function([net.input], [conv_scores])
        # Start visualizations
        motif_extraction(custom_fun, x_data, conv_layer.filters,
                         conv_layer.kernel_size[0], handle, depth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network visualization module.')
    parser.add_argument("model", help='Path to the file')

    parser.add_argument("-d", "--dataset", type=str, default='model',
                        help='Dataset on which the motifs will be generated upon. Write "model" to infer' \
                             'automatically from model.')

    args = parser.parse_args()

    kwargs = {'filename': args.model,
              'data_id': args.dataset}

    main(**kwargs)

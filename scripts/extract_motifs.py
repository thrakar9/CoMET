#!/usr/bin/env python
# coding=utf-8
import math
import os
import sys

import keras.backend as K
import pandas as pd
from absl import flags
from keras.utils import Sequence

from evolutron.engine import load_model
from evolutron.extra_layers import custom_layers
from evolutron.motifs import motif_extraction
from evolutron.tools import preprocess_dataset

flags.DEFINE_string('infile', '', 'The input file. THe supported format currently is a TSV output from UniProt.', )
flags.DEFINE_string('model_file', '', 'The model file for the model to use for predictions.')
flags.DEFINE_string("output_dir", os.path.expandvars(os.curdir), 'The directory to store CoMET output data.')
flags.DEFINE_enum("motifs_filetype", 'txt+png', ['png', 'pdf', 'txt', 'txt+pdf', 'txt+png'],
                  'Choose between different file types to save the extracted motifs from CoMET.'
                  'A typical workflow for subsequent analysis would be to extract the motifs as text files (txt) and'
                  'then use the tool sites2meme to transform them to MEME format and submit for search in MAST.')

FLAGS = flags.FLAGS

try:
    FLAGS(sys.argv)
except flags.Error as e:
    print(e)
    print(FLAGS)
    sys.exit(1)


class UniProtSequence(Sequence):

    def __init__(self, dataframe, batch_size, max_dim):
        self.x = dataframe
        self.batch_size = batch_size
        self.max_dim = max_dim

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.process_batch(batch_x)

    def process_batch(self, batch):
        return preprocess_dataset(batch.Sequence, padded=True, max_aa=self.max_dim, min_aa=self.max_dim)


def extract_motifs(model, proteins, handle):
    # TODO: refactor motif extraction
    # data_gen = UniProtSequence(proteins, 100, model.input_shape[1])

    for depth, conv_layer in enumerate(model.get_conv_layers()):
        conv_scores = conv_layer.output
        # Compile function that spits out the outputs of the correct convolutional layer
        boolean_mask = K.any(K.not_equal(model.input, 0.0), axis=-1, keepdims=True)
        conv_scores = conv_scores * K.cast(boolean_mask, K.floatx())

        # motif_extractor = Model([model.input], [conv_scores])
        custom_fun = K.function([model.input], [conv_scores])

        x_data = preprocess_dataset(proteins.Sequence, padded=True, max_aa=model.input_shape[1],
                                    min_aa=model.input_shape[1])

        # Start motif extraction
        motif_extraction(custom_fun, x_data, conv_layer.filters, conv_layer.kernel_size[0], handle, depth,
                         filetype=FLAGS.motifs_filetype)


def main():
    try:
        model = load_model(FLAGS.model_file, custom_objects=custom_layers)
    except Exception as e:
        print('Unable to load model.')
        print(e)
        sys.exit(1)

    model_key = FLAGS.model_file.split('/')[-1].split('.')[0]

    try:
        proteins = pd.read_csv(FLAGS.infile, sep='\t')
    except Exception as e:
        print('Unable to read input protein dataset.')
        print(e)
        sys.exit(1)

    dataset_key = FLAGS.infile.split('/')[-1].split('.')[0]

    output_folder = os.path.join(FLAGS.output_dir, 'motifs', dataset_key, model_key)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    extract_motifs(model, proteins, output_folder)


if __name__ == "__main__":
    main()

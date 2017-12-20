# coding=utf-8

import math
import os
import sys

import numpy as np
import pandas as pd
from absl import flags
from keras.utils import Sequence

from evolutron.engine import Model, load_model
from evolutron.extra_layers import custom_layers
from evolutron.tools import preprocess_dataset

flags.DEFINE_string('infile', '', 'The input file. THe supported format currently is a TSV output from UniProt.', )
flags.DEFINE_string('model_file', '', 'The model file for the model to use for predictions.')
flags.DEFINE_string("data_dir", '', 'The directory to store CoMET output data.')
flags.DEFINE_string('output_file', None, 'The output file in which to store the hits.')

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


def calculate_embeddings(model, proteins, embed_foldername):
    data_gen = UniProtSequence(proteins, 100, model.input_shape[1])

    code_layer = [layer for layer in model.get_all_layers() if layer.name.find('FCEnc') == 0][-1]

    embedder = Model(inputs=[model.input], outputs=[code_layer.output])

    emb = embedder.predict_generator(data_gen, use_multiprocessing=True, workers=4)

    if not os.path.exists(embed_foldername):
        os.makedirs(embed_foldername)

    if not FLAGS.output_file:
        np.savez(os.path.join(embed_foldername, 'embeddings.npz'), emb)
    else:
        np.savez(os.path.join(embed_foldername, FLAGS.output_file), emb)


def main():
    try:
        proteins = pd.read_csv(FLAGS.infile, sep='\t')
    except Exception as e:
        print('Unable to read input protein dataset.')
        print(e)
        sys.exit(1)

    dataset_key = FLAGS.infile.split('/')[-1].split('.')[0]

    try:
        model = load_model(FLAGS.model_file, custom_objects=custom_layers)
    except Exception as e:
        print('Unable to load model.')
        print(e)
        sys.exit(1)

    model_key = FLAGS.model_file.split('/')[-1].split('.')[0]

    if not FLAGS.output_file:
        embed_foldername = os.path.join(FLAGS.data_dir, 'embeddings', model_key, dataset_key)
    else:
        embed_foldername = os.path.join(FLAGS.data_dir, 'embeddings')

    if not os.path.isfile(embed_foldername):
        calculate_embeddings(model, proteins, embed_foldername)


if __name__ == '__main__':
    main()

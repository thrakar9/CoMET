import math
import os
import sys

import pandas as pd
from absl import flags
from keras.utils import Sequence

from evolutron.engine import load_model
from evolutron.extra_layers import custom_layers
from evolutron.tools import preprocess_dataset

flags.DEFINE_string('infile', '', 'The input file. THe supported format currently is a TSV output from UniProt.', )
flags.DEFINE_string('model_file', '', 'The model file for the model to use for predictions.')
flags.DEFINE_string('output_file', '', 'The model file for the model to use for predictions.')

flags.DEFINE_string("data_dir", '', 'The directory to store CoMET output data.')

FLAGS = flags.FLAGS

try:
    FLAGS(sys.argv)
except flags.Error as e:
    print(e)
    print(FLAGS)
    sys.exit(1)


class HSTSequence(Sequence):

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


def main():

    m = load_model(FLAGS.model_file, custom_objects=custom_layers)

    data = pd.read_csv(FLAGS.infile, sep='\t')

    hst_gen = HSTSequence(data, 100, m.input_shape[1])

    preds = m.predict_generator(hst_gen, use_multiprocessing=True, workers=4)

    data['comet_predictions'] = preds

    output = data[(data['comet_predictions'] > 0.5)]

    output.to_csv(os.path.join(FLAGS.data_dir, 'comet/preds', FLAGS.output_file + '.csv'))


if __name__ == "__main__":
    main()

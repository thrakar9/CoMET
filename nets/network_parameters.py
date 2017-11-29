from absl import flags

flags.DEFINE_multi_integer("filters", 100, 'Number of filters in the convolutional layers.', lower_bound=0)
flags.DEFINE_multi_integer("filter_length", 25, 'Size of filters in the first convolutional layer.', lower_bound=0)
flags.DEFINE_integer("n_conv_layers", 1, 'The number of Convolutional layers.', lower_bound=0)
flags.DEFINE_integer("n_fc_layers", 1, 'The number of Fully-Connected layers.', lower_bound=0)

flags.DEFINE_enum("optimizer", 'nadam', ['adam', 'nadam', 'rmsprop', 'sgd', 'adadelta', 'adagrad'],
                   "The optimizer to use for training.")
flags.DEFINE_float("learning_rate", 0.002, 'The learning rate for the optimizer.')

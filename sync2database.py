import os
import sys
import glob

import numpy as np
import pymysql

try:
    from evolutron.tools import Handle
except ImportError:
    sys.path.insert(0, os.path.abspath('..'))
    from evolutron.tools import Handle

history = glob.iglob('models/**/*.history.npz', recursive=True)

con = pymysql.connect(host='localhost',
                      user='evolutron',
                      password='Evolutron731021!',
                      db='evolutron',
                      charset='utf8mb4',
                      cursorclass=pymysql.cursors.DictCursor)

columns = (
    'model',
    'filters',
    'filter_size',
    'epochs',
    'data_id',
    'handle',
    'fc_layers',
    'conv_layers',
    'loss',
    'acc',
    'val_loss',
    'val_acc',
    'metrics'
)

with con.cursor() as cur:
    for f in history:
        # print(f)
        handle = Handle.from_filename(f)

        with np.load(f) as his:
            try:
                try:
                    loss = his['loss'][-1]
                except KeyError:
                    loss = his['train_loss'][-1]

                val_loss = his['val_loss'][-1]

                try:
                    acc = his['train_acc'][-1]
                except KeyError:
                    acc = his['mean_cat_acc'][-1]

                try:
                    val_acc = his['val_mean_cat_acc'][-1]
                except KeyError:
                    val_acc = his['val_acc_mem'][-1]
            except IndexError as e:
                print(f)
                loss = None
                acc = None
                val_loss = None
                val_acc = None
                raise

        values = (
            handle.model,
            handle.filters,
            handle.filter_size,
            handle.epochs,
            handle.dataset,
            str(handle).split('.')[0],
            handle.n_fc,
            handle.n_convs,
            loss,
            acc,
            val_loss,
            val_acc,
            str(his.files).replace("'", "")
        )

        sql = "INSERT IGNORE INTO `comet_models` ({0}) VALUES {1}".format(",".join(columns), str(values))
        print(sql)
        cur.execute(sql)

con.commit()

# TODO: implement check for deleted files

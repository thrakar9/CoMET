import os
import sys
import glob
import psycopg2

import numpy as np

sys.path.insert(0, os.path.abspath('../Evolutron'))
from evolutron.tools import Handle


def filename_exists(cur, filename):
    cur.execute("SELECT filename FROM comet_models WHERE filename = %s", (filename,))
    return cur.fetchone() is not None


def update_db():
    history = glob.iglob('models/**/*.history.npz', recursive=True)

    conn = psycopg2.connect(host='localhost',
                            user='evomaster',
                            password='evolutron',
                            database='evolutron_comet')

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
    values = []
    for f in history:
        handle = Handle.from_filename(f)

        with np.load(f) as his:
            try:
                loss = min(his['loss'])

                val_loss = min(his['val_loss'])

                acc = max(his['mean_cat_acc'])

                val_acc = max(his['val_mean_cat_acc'])

                values.append((
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
                    val_acc))
            except Exception as e:
                print(e)
                print(f)
                raise Exception

    with conn.cursor() as cur:

        args_str = b','.join(
                cur.mogrify("(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", x) for x in values if not filename_exists(cur, x[0]))

        cur.execute(
                b"INSERT INTO comet_models (model, filters, kernel_size, epochs, dataset, filename, "
                b"fc_layers, conv_layers, loss, acc, val_loss,val_acc) VALUES " + args_str +
                b" ON CONFLICT (filename) DO UPDATE SET val_loss = EXCLUDED.val_loss, val_acc = EXCLUDED.val_acc")


        conn.commit()

    conn.close()

        # TODO: implement check for deleted files

if __name__ == '__main__':
    update_db()

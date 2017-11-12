# coding=utf-8
import json

import h5py


def get_weights_from_hdf5_group(hfg):
    layer_names = [n.decode('utf8') for n in hfg.attrs['layer_names']]
    weight_dict = dict()
    for name in layer_names:
        g = hfg[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name].value.tolist() for weight_name in weight_names]
        weight_names = map(lambda x: x.split('/')[-1].replace(':0', ''), weight_names)
        weight_dict[name] = dict(zip(weight_names, weight_values))

    return weight_dict


filename = 'models/acetyl/100_30_34_1_1_DeepCoFAM.model'

with h5py.File(filename, mode='r') as hf:
    model_config = hf.attrs['model_config'].decode('utf8')
    model_weights = json.dumps(get_weights_from_hdf5_group(hf['model_weights']))

with open('test.model.json', 'w') as f:
    f.write("""{"config": %s, "weights": %s}""" % (model_config, model_weights))

# coding=utf-8

import argparse
import os
import sys

import h5py
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import model_from_json
from scipy.cluster.hierarchy import fclusterdata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from evolutron.engine import DeepTrainer
    from evolutron.templates import custom_layers
    from evolutron.tools import aa2hot, Handle, load_dataset, shape, file_db
    from evolutron.tools.data_tools import pad_or_clip_seq
except ImportError:

    sys.path.insert(0, os.path.abspath('../Evolutron'))
    from evolutron.engine import DeepTrainer
    from evolutron.templates import custom_layers
    from evolutron.tools import aa2hot, Handle, load_dataset, shape, file_db
    from evolutron.tools.data_tools import pad_or_clip_seq


def calculate_embeddings(model, proteins):
    handle = Handle.from_filename(model)

    # Load model architecture, build model and then load trained weights.
    with h5py.File(model) as hf:
        model_config = hf.attrs['model_config'].decode('utf8')
    net = DeepTrainer(model_from_json(model_config, custom_objects=custom_layers))
    net.load_all_param_values(model)

    x_data = proteins.sequence.apply(aa2hot).tolist()
    max_aa = net.input._keras_shape[1]
    x_data = np.asarray([pad_or_clip_seq(x, max_aa) for x in x_data])

    code_layer = [layer for layer in net.get_all_layers() if layer.name.find('FCEnc') == 0][-1]

    embed_fun = K.function(inputs=[net.input], outputs=[code_layer.output])

    emb = np.asarray([embed_fun([[x]]) for x in x_data]).squeeze()

    foldername = 'embeddings/' + handle.data_id + '/'
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    np.savez(foldername + handle.filename.split('.')[0] + '.embed.npz', emb)
    return emb


def main(model, tsne=True, html=False, pdf=False, pca=False):
    handle = Handle.from_filename(model)

    try:
        proteins = pd.read_hdf(file_db[handle.data_id].split('.')[0] + '.h5', 'raw_data')
    except KeyError:
        proteins = pd.read_hdf('/data/datasets/' + handle.data_id + '.h5', 'raw_data')

    try:
        with np.load('embeddings/' + handle.data_id + '/' + handle.filename.split('.')[0] + '.embed.npz') as f:
            emb = f['arr_0']
        print('Loaded embeddings')

    except IOError:
        emb = calculate_embeddings(model, proteins)
        print('Generated embeddings')

    embeddings = pd.DataFrame(emb, index=proteins.index, columns=['Emb{}'.format(i) for i in range(emb.shape[1])])
    assert (len(embeddings) == len(proteins))

    # proteins = proteins.sample(n=max).reset_index(drop=True)

    fam = proteins['family'].astype('category')

    all_fams = fam.cat.categories.tolist()

    vocabulary = fam.tolist()

    low_values_ind = emb < 0.5
    high_values_ind = emb > 0.5
    emb[low_values_ind] = .0
    emb[high_values_ind] = 1.
    if tsne:
        try:
            with np.load('embeddings/' + handle.data_id + '/' + handle.filename.split('.')[0] + '.Y.embed.npz') as f:
                Y = f['arr_0']
            print('Loaded TSNE of embeddings')
        except FileNotFoundError:
            tsne = TSNE(n_components=2, verbose=2, metric='hamming', n_iter=1000, n_iter_without_progress=200)
            Y = tsne.fit_transform(emb[np.random.choice(emb.shape[0], 20000, replace=False), :])
            np.savez('embeddings/' + handle.data_id + '/' + handle.filename.split('.')[0] + '.Y.embed.npz', Y)
            print('Generated TSNE of embeddings')

    if pca:
        try:
            with np.load('embeddings/' + handle.data_id + '/' + handle.filename.split('.')[0] + '.PCA.embed.npz') as f:
                pca = f['arr_0']
            print('Loaded PCA of embeddings')
        except FileNotFoundError:
            pca_model = PCA(n_components=min(emb.shape[1], 30))
            pca = pca_model.fit_transform(emb)
            np.savez('embeddings/' + handle.data_id + '/' + handle.filename.split('.')[0] + '.PCA.embed.npz', pca)
            print('Generated PCA of embeddings')

        try:
            with np.load('embeddings/' + handle.data_id + '/' + handle.filename.split('.')[0] +
                                 '.Y_PCA.embed.npz') as f:
                Y_PCA = f['arr_0']
            print('Loaded TSNE of PCA of embeddings')

        except FileNotFoundError:
            tsne = TSNE(n_components=2, random_state=0, verbose=1)
            Y_PCA = tsne.fit_transform(pca)
            np.savez('embeddings/' + handle.data_id + '/' + handle.filename.split('.')[0] + '.Y_PCA.embed.npz', Y_PCA)
            print('Generated TSNE of PCA of embeddings')

    print('done')

    # # Clustering
    table = pd.concat([proteins, embeddings], axis=1)
    # size = 1  # sample size
    # replace = False  # with replacement
    #
    # def fn(obj):
    #     return obj.loc[np.random.choice(obj.index, size, replace), :]

    sample = table.groupby('family', as_index=True).mean()
    print(len(sample))
    if len(sample) > 1000:
        sample = sample.sample(n=500)
    print(len(sample))

    emb_fam = sample[['Emb{}'.format(i) for i in range(emb.shape[1])]].as_matrix()
    from scipy.cluster.hierarchy import linkage
    from scipy.cluster.hierarchy import to_tree
    z = linkage(emb_fam, 'weighted', metric='hamming')

    def getNewick(node, newick, parentdist, leaf_names):
        if node.is_leaf():
            return "%s:%.2f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
        else:
            if len(newick) > 0:
                newick = "):%.2f%s" % (parentdist - node.dist, newick)
            else:
                newick = ");"
            newick = getNewick(node.get_left(), newick, node.dist, leaf_names)
            newick = getNewick(node.get_right(), ",%s" % (newick), node.dist, leaf_names)
            newick = "(%s" % (newick)
            return newick

    tree = to_tree(z, False)
    names = sample.index.tolist()
    newick = getNewick(tree, "", tree.dist, names)

    labels = fclusterdata(emb, t=.5, metric='hamming')
    if html:
        proteins['labels'] = pd.Series(labels, index=proteins.index)
        proteins['tsn1'] = pd.Series(Y[:, 0], index=proteins.index)
        proteins['tsn2'] = pd.Series(Y[:, 1], index=proteins.index)
        proteins['fam'] = proteins['fam'].str.replace('family', '')
        server_media = '/Users/thrakar9/Desktop/repo/EvolutronServer/media'
        proteins.to_csv(server_media + '/embeds/' + handle.data_id + '_' + handle.filename.split('.')[0] + '.csv',
                        columns=['protein_names', 'fam', 'sup', 'sub', 'tsn1', 'tsn2', 'labels'])

        with open(server_media + '/embeds/' + handle.data_id + '_' + handle.filename.split('.')[0] + '.tree', 'w') as f:
            f.write(newick)


    # colors = []
    # for voc in vocabulary:
    #
    #     if voc.find('RING') > 0:
    #         colors.append('r')
    #     elif voc.find('PHD') > 0:
    #         colors.append('g')
    #     elif voc.find('polymerase') > 0:
    #         colors.append('y')
    #     else:
    #         colors.append('b')

    # groups = proteins.groupby('labels')
    if pdf:
        colors = proteins['fam'].cat.codes

        fig, ax = plt.subplots()
        ax.scatter(Y[:, 0], Y[:, 1], c=colors)
        ax.set_title(handle.data_id + ' #fam {0}'.format(len(all_fams)))
        # plt.xlim([-10, 10])
        # plt.ylim([-10, 10])
        # for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        #     plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        fig.savefig('show/{0}_{1}_{2}'.format(handle.data_id, handle.filters, handle.filter_size) +
                    '_Y.pdf', dpi=500, facecolor='w', edgecolor='w')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning protein embeddings')
    parser.add_argument("model", help='Path to the file')
    parser.add_argument("--html", action='store_true')
    parser.add_argument("--pdf", action='store_true')

    args = parser.parse_args()

    kwargs = {'model': args.model,
              'html': args.html,
              'pdf': args.pdf}

    main(**kwargs)

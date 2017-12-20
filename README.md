# CoMET - **Co**nvolutional **M**otif **E**mbeddings **T**ools

![models](https://s3.amazonaws.com/comet-media/github_models.png)

As the number of available protein sequences increases exponentially, existing motif discovery tools, are reaching their limits in required time and effort. To overcome these limitations, we introduce motif embeddings, a hierarchical decomposition of protein sequences into motifs. To learn the embeddings, we developed a series of state-of-the-art deep learning tools: **CoMET**. 

At the core of CoMET is a Deep Convolutional Encoder. Two exemplary network architectures that take as input motif embeddings, are: **CoDER**, which extracts motif embeddings from a set of protein sequences, where no other information is available, in an unsupervised way, and : **CoFAM**, which allows us to learn motif embeddings from protein sequences in a supervised learning setup, when, for example, protein family information is available.

## Install CoMET

### Clone Evolutron and add it to the path
```
git clone https://github.com/mitmedialab/Evolutron.git $HOME/.evolutron
echo "export PATH=$HOME/.evolutron/:$PATH"
```
**WARNING: Evolutron is still under rapid development and CoMET is always updated to use the latest commit on master. Please pull the Evolutron repo immediatly after any update on the CoMET repo** 

### Clone CoMET
```
git clone https://github.com/thrakar9/CoMET.git
```

## How to train CoMET models

CoMET options are parsed using [ABSL flags (former gflags)](https://github.com/abseil/abseil-py). 
You can control them by command line arguments or configuration files (see example/example.conf).

### Unsupervised Motif Extraction (CoDER) Example:
```shell
python CoMET.py --flagfile example/example.conf --mode CoDER
```

### Family Classification (CoFAM) Example:
```shell
python CoMET.py --flagfile example/example.conf --mode CoFAM
```
   
### Binary Classification for Homology Search (CoHST) Example:
```shell
python CoMET.py --flagfile example/example.conf --mode CoHST
```

### Output Files Description
CoMET generates output files in the directory given by the `data_dir` flag.

Each experiment (run) is assigned a unique random ID, by which you can identify the network architecture, 
the trained model, the training history and the flags used.

The generated output file structure is the following:

* models/
  * ID.model: The saved Evolutron (Keras) model.
  * ID.flags: The flags' values dictionary pickled with dill.
  * ID.arch: The architecture of the network saved as a JSON config file.
  * ID.history.npz: The training history saved as a Numpy compressed file.

## Useful Scripts
Here is a list of scripts that you can use to apply CoMET models to protein sequence datasets.

### Generate Embeddings from protein sequences
This script works with any type of trained CoMET model, and produces the motif-embeddings of the input protein sequences.

```shell
python scripts/generate_embeddings.py --infile /path/to/dataset.tsv --model_file=/path/to/model.model [--output_file output_filename]
```
The output is saved at `embeddings/dataset/{model_ID}/{output_file}.npz`.
      
### Extract motifs from protein sequences
This script works with any type of trained CoMET model, and extracts sequence motifs from a set of input protein sequences, by looking at the receptive fields of the convolutional neurons.

```shell
python scripts/extract_motifs.py --infile /path/to/dataset.tsv --model_file=/path/to/model.model [--output_dir output_foldername]
```

The generated output file structure is the following:

* `{output_dir}/motifs/dataset/{model_ID}/`
  * `1/`: The motifs of the first (closest to the input) convolutional layer.
    * `XX_YY.png`: The motif extracted from neuron at position XX, from YY number of protein sequences.
    * `XX_YY.txt`: The YY sequence patterns that activated the neuron XX in order to generate the motif.
  * `2/`, `3/`, ...: The position of the next convolutional layers in the same structure as above.

### Search for homologous protein sequences
This script works with CoHST trained models, and scans a set of input protein sequences to identify sequence homologs to the protein dataset that was used as positive when training the model.

```shell
python scripts/search_for_homologs.py --infile /path/to/dataset --model_file=/path/to/model.model [--output_file output_filename]
```
The output is saved at `homologs/dataset/{model_ID}/{output_file}.npz`.

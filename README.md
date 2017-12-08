# CoMET - **Co**nvolutional **M**otif **E**mbeddings **T**ools

![models](https://s3.amazonaws.com/comet-media/github_models.png)

As the number of available protein sequences increases exponentially, existing motif discovery tools, are reaching their limits in required time and effort. To overcome these limitations, we introduce motif embeddings, a hierarchical decomposition of protein sequences into motifs. To learn the embeddings, we developed a series of state-of-the-art deep learning tools: **CoMET**. 

At the core of CoMET is a Deep Convolutional Encoder. Two exemplary network architectures that take as input motif embeddings, are: **CoDER**, which extracts motif embeddings from a set of protein sequences, where no other information is available, in an unsupervised way, and : **CoFAM**, which allows us to learn motif embeddings from protein sequences in a supervised learning setup, when, for example, protein family information is available.

## Install CoMET

### Clone Evolutron and add it to the path
```
git clone https://github.com/mitmedialab/Evolutron.git ~/.evolutron
echo "export PATH=~/.evolutron/:$PATH"
```

### Clone CoMET
```
git clone https://github.com/mitmedialab/CoMET.git
```

## How to train CoMET models

CoMET options are parsed using [ABSL flags (former gflags)](https://github.com/abseil/abseil-py). 
You can controll them by command line arguments or configuration files (see example/example.conf).

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

[ TODO: describe motif extraction ]

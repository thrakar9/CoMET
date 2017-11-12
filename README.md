# CoMET - **Co**nvolutional **M**otif **E**mbeddings **T**ools

![models](https://s3.amazonaws.com/comet-media/github_models.png)

As the number of available protein sequences increases exponentially, existing motif discovery tools, are reaching their limits in required time and effort. To overcome these limitations, we introduce motif embeddings, a hierarchical decomposition of protein sequences into motifs. To learn the embeddings, we developed a series of state-of-the-art deep learning tools: **CoMET**. 

At the core of CoMET is a Deep Convolutional Encoder. Two exemplary network architectures that take as input motif embeddings, are: **CoDER**, which extracts motif embeddings from a set of protein sequences, where no other information is available, in an unsupervised way, and : **CoFAM**, which allows us to learn motif embeddings from protein sequences in a supervised learning setup, when, for example, protein family information is available.

## Install CoMET

### Clone Evolutron and add it to the path
```
git clone https://github.mit.edu/karydis/Evolutron.git ~/.evolutron
echo "export PATH=~/.evolutron/:$PATH"
```

### Clone CoMET
```
git clone https://github.mit.edu/karydis/CoMET.git ~/.evolutron
```

## How to train CoMET models

### Unsupervised Mode (CoDER) Example:
   ```shell
   python CoMET.py -i example/uniprot_cas9.tsv --filters 200 --filter_length 30 -e 200 --conv 1 --fc 1
   ```

### Supervised Mode (CoFAM) Example:
   ```shell
   python CoMET.py --mode family -i example/uniprot_cas9.tsv --filters 200 --filter_length 30 -e 200 --conv 1 --fc 1
   ```

[ describe motif extraction ]

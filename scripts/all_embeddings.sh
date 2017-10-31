#!/usr/bin/env bash
source ${HOME}/.bashrc
workon py3Evolutron

dataset=$1
echo "Calculating all embeddings for" ${dataset}

for f in models/${dataset}/*.history.npz ;do
    THEANO_FLAGS="device=cpu" python embed.py ${f%%.*}.model --html
done


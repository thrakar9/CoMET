#!/usr/bin/env bash
source ${HOME}/.bashrc
workon py3Evolutron

dataset=crispr
echo "Calculating all motifs for" ${dataset}

for f in models/${dataset}/*.model ;do
    t=${f%.*}
    if [ ! -d motifs/${t#models} ]; then
        python visualize_motifs.py ${f}
    fi
done

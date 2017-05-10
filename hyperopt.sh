#!/usr/bin/env bash
#PBS -q batch
#PBS -A thras
#PBS -N CoMET3
#PBS -l ncpus=16,gpus=1,walltime=01:00:00
#PBS -k o
#PBS -j oe

cd /home/thras/Desktop/CoMET/

for i in 2 5 10 25 50 75; do
    CUDA_VISIBLE_DEVICES=0 python CoMET.py dnabind ${i} 15 --conv 2 -e 200  > logs/log0.txt
done &

for i in 1 5 10 25 50 100; do
    CUDA_VISIBLE_DEVICES=1 python CoMET.py dnabind ${i} 10 --conv 3 -e 200  > logs/log0.txt
done &

for i in 1 2 10 25 75 100; do
    CUDA_VISIBLE_DEVICES=2 python CoMET.py dnabind ${i} 5 --conv 4 -e 200 > logs/log0.txt
done &

for i in 1 2 5 50 75 100; do
    CUDA_VISIBLE_DEVICES=3 python CoMET.py dnabind ${i} 3 --conv 5 -e 200  > logs/log0.txt
done &

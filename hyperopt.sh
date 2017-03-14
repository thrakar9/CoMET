#!/usr/bin/env bash

THEANO_FLAGS="device=gpu0" python CoMET.py mycoplasma 100 50 -e 200 --conv 1 --fc 2 --rate .01 > logs/log0.txt &
THEANO_FLAGS="device=gpu1" python CoMET.py mycoplasma 100 25 -e 200 --conv 2 --fc 2 --rate .01 > logs/log1.txt &
THEANO_FLAGS="device=gpu2" python CoMET.py mycoplasma 100 5 -e 200 --conv 3 --fc 2 --rate .01 > logs/log2.txt &
#THEANO_FLAGS="device=gpu3" python CoMET.py crispr 200 30 --mode family -e 200 --conv 1 --rate .01 > logs/log3.txt &
#THEANO_FLAGS="device=gpu4" python CoMET.py dnabind 100 100 -e 200 --conv 1 --rate .01 > logs/log4.txt &
#THEANO_FLAGS="device=gpu5" python CoMET.py hsapiens 256 5 --mode family -e 200 --conv 1 --rate .01 > logs/log5.txt &
#THEANO_FLAGS="device=gpu6" python CoMET.py dnabind 100 20 --mode family -e 200 --conv 3 --rate .01 > logs/log6.txt &
#THEANO_FLAGS="device=gpu7" python CoMET.py dinbind 100 20 --mode family -e 200 --conv 3 --rate .01 > logs/log7.txt &

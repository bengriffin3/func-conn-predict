#!/bin/bash
#SBATCH -J edg-pre-50-4-1220-icov-icov-pc-1-8-10-1
#SBATCH -o logs/edg-pre-50-4-1220-icov-icov-pc-1-8-10-1.out
#SBATCH -e logs/edg-pre-50-4-1220-icov-icov-pc-1-8-10-1.err
#SBATCH -p short
source activate osld
python ../scripts/02_netmats_edge_prediction.py 50 1220 icov icov 4 pc --run 1 --n_states 8 --trans_prob_diag 10 --model_mean

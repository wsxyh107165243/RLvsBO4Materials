#!/bin/bash
#SBATCH -J python_sma_grad_inner_argmax
#SBATCH -p node
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -n 48

module load intel18u4

date
conda run -n SMA_grad_inner_argmax_310 python3 experiments/run_batch_test.py
date
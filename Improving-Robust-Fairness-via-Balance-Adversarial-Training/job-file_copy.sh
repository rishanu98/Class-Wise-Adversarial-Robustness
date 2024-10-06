#!/bin/bash
#SBATCH --job-name=my_gpu_job_1
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

python train_copy.py

#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --time=23:59:59
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

python frlrmrw.py

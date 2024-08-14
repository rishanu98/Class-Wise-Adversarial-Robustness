#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --time=23:59:59
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

python train_copy.py --mode 'AT' --fname 'AT_CCM' --ccm --lambda-1 0.5 --threshold 0.2 
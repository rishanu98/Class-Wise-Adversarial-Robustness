#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

python train.py --mode 'AT' --fname 'AT_CCM_Normalized' --epsilon 0.0313 --ccm --lambda-1 0.5 --threshold 0.2 --epochs 100
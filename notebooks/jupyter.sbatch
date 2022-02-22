#!/bin/bash

#SBATCH -o /home/icb/%u/slurm_folder/slurm_jupyter.job

#SBATCH -e /home/icb/%u/slurm_folder/slurm_jupyter.job

#SBATCH -J jupyter

#SBATCH -p gpu_p

#SBATCH --gres=gpu:2

#SBATCH --qos=gpu

#SBATCH -c 4

#BATCH --mem=90G

#SBATCH -t 10:00:00

#SBATCH --nice=10000


source ./run_jupyter.sh


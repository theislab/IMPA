#!/bin/bash

#SBATCH -o /home/icb/%u/slurm_folder/dmso.job

#SBATCH -e /home/icb/%u/slurm_folder/dmso.job

#SBATCH -J dmso

#SBATCH -p gpu_p

#SBATCH --qos=gpu

#SBATCH --gres=gpu:1

#SBATCH -c 4

#BATCH --mem=90G

#SBATCH --nice=10000

source $HOME/.bashrc


conda activate imgCPA

python ./dmso.py

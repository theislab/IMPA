#!/bin/bash

#SBATCH -o /home/icb/%u/slurm_jupyter_%j.job

#SBATCH -e /home/icb/%u/slurm_jupyter_%j.job

#SBATCH -J jupyter

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 10:00:00

#SBATCH --nice=10000


source ./run_jupyter.sh


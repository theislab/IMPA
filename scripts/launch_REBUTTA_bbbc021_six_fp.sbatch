#!/bin/bash

#SBATCH -o ./logs/fp_large_all.out

#SBATCH -e ./logs/fp_large_all.err

#SBATCH -J fp_large_all

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA_try

python ../IMPA/main_hydra.py config=REBUTTAL_bbbc021_large_all.yaml
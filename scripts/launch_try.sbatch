#!/bin/bash

#SBATCH -o ./logs/out_try.out

#SBATCH -e ./logs/out_try.err

#SBATCH -J bbbc021_try

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA_try

python ../IMPA/main_hydra.py 
 
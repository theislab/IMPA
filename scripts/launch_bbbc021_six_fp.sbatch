#!/bin/bash

#SBATCH -o ./logs/fp.out

#SBATCH -e ./logs/fp.err

#SBATCH -J fp

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA_try

python ../IMPA/main_hydra.py config=bbbc021_six_morgan_fp.yaml
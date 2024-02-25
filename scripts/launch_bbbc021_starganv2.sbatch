#!/bin/bash

#SBATCH -o ./logs/output_bbbc021_starganv2.out

#SBATCH -e ./logs/output_bbbc021_starganv2.err

#SBATCH -J bbbc021_starganv2

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA_try

python ../IMPA/main_hydra.py config=bbbc021_starganv2_all_retrain.yaml config.project=bbbc021_all_retrain_starganv2

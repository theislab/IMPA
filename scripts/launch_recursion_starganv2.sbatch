#!/bin/bash

#SBATCH -o ./logs/output_recursion_stargan.txt

#SBATCH -e ./logs/error_recursion_stargan.txt

#SBATCH -J recursion_stargan

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA_try

python ../IMPA/main_hydra.py config=rxrx1_retrain_starganv2.yaml 
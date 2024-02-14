#!/bin/bash

#SBATCH -o ./logs/output_bbbc021_all.txt

#SBATCH -e ./logs/error_bbbc021_all.txt

#SBATCH -J bbbc021

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA

cd /path/to/IMPA/folder

python main_not_seml.py training_configs/bbbc021_all_config.yaml

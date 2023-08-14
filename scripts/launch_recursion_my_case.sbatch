#!/bin/bash

#SBATCH -o ./logs/output_recursion.txt

#SBATCH -e ./logs/error_recursion.txt

#SBATCH -J recursion

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate imgCPA

cd /home/icb/alessandro.palma/environment/IMPA/IMPA/IMPA

python main_not_seml.py ../training_configs/recursion_config.yaml
 
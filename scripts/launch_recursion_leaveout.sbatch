#!/bin/bash

#SBATCH -o ./logs/output_recursion_leaveout.txt

#SBATCH -e ./logs/error_recursion_leaveout.txt

#SBATCH -J recursion_leaveout

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu_normal

#SBATCH -c 4

#SBATCH --mem=150G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA

cd /path/to/IMPA/folder

python main_not_seml.py training_configs/recursion_config.yaml
 
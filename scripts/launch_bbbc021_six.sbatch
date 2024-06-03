#!/bin/bash

#SBATCH -o ./logs/output_bbbc021_six.out

#SBATCH -e ./logs/error_bbbc021_six.err

#SBATCH -J bbbc021

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA_try

cd ..

python IMPA/main_hydra.py config=bbbc021_six
 
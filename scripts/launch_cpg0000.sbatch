#!/bin/bash

#SBATCH -o ./logs/output_cpg0000.out

#SBATCH -e ./logs/error_cpg0000.err

#SBATCH -J cpg0000

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
export CUDA_LAUNCH_BLOCKING=1
conda activate IMPA_try

cd ..

python IMPA/main_hydra.py config=cpg0000
 
 
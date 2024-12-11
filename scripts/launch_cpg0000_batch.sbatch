#!/bin/bash

#SBATCH -o ./logs/output_cpg0000_batch.out

#SBATCH -e ./logs/error_cpg0000_batch.err

#SBATCH -J cpg0000_batch

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu_long

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 3-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA_try

cd ..

python IMPA/main_hydra.py config=cpg0000_batch
 
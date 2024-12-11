#!/bin/bash

#SBATCH -o ./logs/output_rxrx1_batch.out

#SBATCH -e ./logs/error_rxrx1_batch.err

#SBATCH -J rxrx1_batch

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

python IMPA/main_hydra.py config=rxrx1_batch
 
#!/bin/bash

#SBATCH -o ./logs/det_cpg0000_batch.out

#SBATCH -e ./logs/det_cpg0000_batch.err

#SBATCH -J det_cpg0000_batch

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem 150G

#SBATCH -c 4

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA_try

python ../IMPA/main_hydra.py config=rxrx1_batch.yaml config.stochastic=False
# python ../IMPA/main_hydra.py config=cpg0000_batch.yaml 
# python ../IMPA/main_hydra.py config=cpg0000_batch.yaml config.stochastic=False
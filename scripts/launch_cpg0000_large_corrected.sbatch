#!/bin/bash

#SBATCH -o ./logs/cpg0000_separate_ORF_and_CRISPR.out

#SBATCH -e ./logs/cpg0000_separate_ORF_and_CRISPR.err

#SBATCH -J cpg0000_separate_ORF_and_CRISPR

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA_try

python ../IMPA/main_hydra.py config=cpg0000_total_large_transported_separate_ORF_and_CRISPR.yaml 
# python ../IMPA/main_hydra.py config=cpg0000_total_large_transported.yaml hparam_search=optuna --multirun
# python ../IMPA/main_hydra.py config=cpg0000_total_large_transported.yaml config.stochastic=False config.project=cpg0000_total_large_transported_deterministic hparam_search=optuna --multirun
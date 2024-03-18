#!/bin/bash

#SBATCH -o ./logs/multimodal_cond.out

#SBATCH -e ./logs/multimodal_cond.err

#SBATCH -J multimodal_cond

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA_try

# python ../IMPA/main_hydra.py config=cpg0000_multimodal.yaml
python ../IMPA/main_hydra.py config=cpg0000_multimodal_condition_emb.yaml

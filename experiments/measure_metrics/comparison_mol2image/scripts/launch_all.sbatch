#!/bin/bash

#SBATCH -o /home/icb/alessandro.palma/environment/IMPA/IMPA/experiments/measure_metrics/comparison_mol2image/scripts/output.txt

#SBATCH -e /home/icb/alessandro.palma/environment/IMPA/IMPA/experiments/measure_metrics/comparison_mol2image/scripts/error.txt

#SBATCH -J total_evaluation

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA_try

cd /home/icb/alessandro.palma/environment/IMPA/IMPA

python /home/icb/alessandro.palma/environment/IMPA/IMPA/experiments/measure_metrics/comparison_mol2image/compute_scores_ood.py 

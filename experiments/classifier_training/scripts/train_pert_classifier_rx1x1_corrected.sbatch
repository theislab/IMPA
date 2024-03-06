#!/bin/bash

#SBATCH -o ./logs/train_classifier_rxrx1_corrected.out

#SBATCH -e ./logs/train_classifier_rxrx1_corrected.err

#SBATCH -J rxrx1_corrected

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

conda activate IMPA_try

cd /home/icb/alessandro.palma/environment/IMPA/IMPA/experiments/classifier_training

python train_classifier.py --args_path=/home/icb/alessandro.palma/environment/IMPA/IMPA/experiments/classifier_training/compound_classifier_rxrx1_transported.yaml
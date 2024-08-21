#!/bin/bash

#SBATCH -o ./logs/metrics_stytr_2.out

#SBATCH -e ./logs/metrics_stytr_2.err

#SBATCH -J stytr_metrics_2

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate diffusion-experiments

python ../launch_metrics_stytr.py \
--yaml_path /home/icb/alessandro.palma/environment/IMPA/IMPA/experiments/measure_metrics/configs/stytr.yaml \
--ckpt_dir  /home/icb/alessandro.palma/environment/IMPA/StyTR-2/experiments \
--ckpt_epoch 30000 \
--dataset_name  bbbc021 \
--model_name StyTR \
--classifier_ckpt_path /lustre/groups/ml01/workspace/alessandro.palma/imCPA_official/experiments/general_experiments/7.train_classifier/checkpoints/checkpoints_6_drugs.ckpt \
--result_path /home/icb/alessandro.palma/environment/IMPA/IMPA/experiments/measure_metrics/results_stytr/results_bbbbc021_2.pkl
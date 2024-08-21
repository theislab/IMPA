#!/bin/bash

#SBATCH -o ./logs/metrics_phendiff.out

#SBATCH -e ./logs/metrics_phendiff.err

#SBATCH -J metrics_phendiff

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate diffusion-experiments

python ../launch_metrics_phendiff.py --yaml_path /home/icb/alessandro.palma/environment/IMPA/IMPA/experiments/measure_metrics/configs/phendiff.yaml \
--ckpt_dir /home/icb/alessandro.palma/environment/IMPA/PhenDiff/examples/examples_training_scripts/experiments/experiment_name/ddm_six/full_pipeline_save \
--ckpt_epoch 0 \
--dataset_name bbbc021 \
--model_name PhenDiff \
--classifier_ckpt_path /lustre/groups/ml01/workspace/alessandro.palma/imCPA_official/experiments/general_experiments/7.train_classifier/checkpoints/checkpoints_6_drugs.ckpt \
--result_path /home/icb/alessandro.palma/environment/IMPA/IMPA/experiments/measure_metrics/results_phendiff/results_bbbbc021_2.pkl 
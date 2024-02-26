#!/bin/bash

#SBATCH -o ./logs/metrics_retrain.out

#SBATCH -e ./logs/metrics_retrain.err

#SBATCH -J metrics_retrain

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA_try

python ../launch_metrics.py --yaml_path /home/icb/alessandro.palma/environment/IMPA/IMPA/config_hydra/config/bbbc021_all_retrain.yaml --ckpt_dir /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/experiments/20240205_347bc76f-e679-42e6-91b7-8fa5482b6474_bbbc021_all_retrain --ckpt_epoch 150 --dataset_name bbbc021 --model_name IMPA --classifier_ckpt_path /lustre/groups/ml01/workspace/alessandro.palma/imCPA_official/experiments/general_experiments/7.train_classifier/checkpoints/checkpoints_all_drugs.ckpt --result_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/results_revaluations/metrics_retrain.pkl

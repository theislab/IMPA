#!/bin/bash

#SBATCH -o ./logs/metrics_stargan.out

#SBATCH -e ./logs/metrics_stargan.err

#SBATCH -J metrics_stargan

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA_try

python ../launch_metrics.py --yaml_path /home/icb/alessandro.palma/environment/IMPA/IMPA/config_hydra/config/bbbc021_starganv2_all_retrain.yaml --ckpt_dir /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/experiments/20240223_44178347-1bf6-499e-ba8e-ec857ac7b910_bbbc021_all_starganv2 --ckpt_epoch 150 --dataset_name bbbc021 --model_name starGANv2 --classifier_ckpt_path /lustre/groups/ml01/workspace/alessandro.palma/imCPA_official/experiments/general_experiments/7.train_classifier/checkpoints/checkpoints_all_mol_drugs.ckpt --result_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/results_revaluations/metrics_starganv2.pkl

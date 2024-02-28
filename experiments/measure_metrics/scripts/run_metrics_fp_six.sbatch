#!/bin/bash

#SBATCH -o ./logs/metrics_six_starganv2_2.out

#SBATCH -e ./logs/metrics_six_starganv2_2.err

#SBATCH -J metrics_six_starganv2_2

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA_try

# python ../launch_metrics.py --yaml_path /home/icb/alessandro.palma/environment/IMPA/IMPA/config_hydra/config/bbbc021_six_starganv2.yaml --ckpt_dir /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/experiments/20240227_db108664-5242-418d-8237-8ff8dc27cd06_bbbc021_six --ckpt_epoch 150 --dataset_name bbbc021 --model_name IMPA --classifier_ckpt_path /lustre/groups/ml01/workspace/alessandro.palma/imCPA_official/experiments/general_experiments/7.train_classifier/checkpoints/checkpoints_6_drugs.ckpt --result_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/results_revaluations/metrics_stargan_six.pkl --leaveout False
python ../launch_metrics.py --yaml_path /home/icb/alessandro.palma/environment/IMPA/IMPA/config_hydra/config/bbbc021_six_starganv2.yaml --ckpt_dir /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/experiments/20240227_db108664-5242-418d-8237-8ff8dc27cd06_bbbc021_six --ckpt_epoch 150 --dataset_name bbbc021 --model_name starGANv2 --classifier_ckpt_path /lustre/groups/ml01/workspace/alessandro.palma/imCPA_official/experiments/general_experiments/7.train_classifier/checkpoints/checkpoints_6_drugs.ckpt --result_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/results_revaluations/metrics_stargan_six2.pkl --leaveout False


# python ../launch_metrics.py --yaml_path /home/icb/alessandro.palma/environment/IMPA/IMPA/config_hydra/config/bbbc021_all_retrain_fp.yaml --ckpt_dir /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/experiments/20240227_aae7c2c6-b1a5-4c2a-851b-0c672502c090_bbbc021_all_retrain_fp --ckpt_epoch 150 --dataset_name bbbc021 --model_name IMPA --classifier_ckpt_path /lustre/groups/ml01/workspace/alessandro.palma/imCPA_official/experiments/general_experiments/7.train_classifier/checkpoints/checkpoints_all_drugs.ckpt --result_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/results_revaluations/metrics_fp.pkl --leaveout True
# python ../launch_metrics.py --yaml_path /home/icb/alessandro.palma/environment/IMPA/IMPA/config_hydra/config/bbbc021_all_retrain_fp.yaml --ckpt_dir /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/experiments/20240227_aae7c2c6-b1a5-4c2a-851b-0c672502c090_bbbc021_all_retrain_fp --ckpt_epoch 150 --dataset_name bbbc021 --model_name IMPA --classifier_ckpt_path /lustre/groups/ml01/workspace/alessandro.palma/imCPA_official/experiments/general_experiments/7.train_classifier/checkpoints/checkpoints_all_drugs.ckpt --result_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/results_revaluations/metrics_fp_quantile.pkl --leaveout True

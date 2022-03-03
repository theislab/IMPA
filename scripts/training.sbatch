#!/bin/bash

#SBATCH -o /home/icb/%u/slurm_folder/training_job.job

#SBATCH -e /home/icb/%u/slurm_folder/training_job.job

#SBATCH -J training_loop

#SBATCH -p gpu_p

#SBATCH --gres=gpu:2

#SBATCH --qos=gpu

#SBATCH -c 4

#BATCH --mem=90G

#SBATCH --nice=10000


source $HOME/.bashrc

echo Launching the training script

conda activate imgCPA

python /home/icb/alessandro.palma/imCPA/main.py --config_file=/home/icb/alessandro.palma/imCPA/configs/config_sigma_vae.json --mode=train
#python /home/icb/alessandro.palma/imCPA/main.py --config_file=/home/icb/alessandro.palma/imCPA/configs/config_glow.json --mode=train


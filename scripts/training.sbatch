#!/bin/bash

#SBATCH -o /home/icb/%u/slurm_folder/training_job.job

#SBATCH -e /home/icb/%u/slurm_folder/training_job.job

#SBATCH -J training_loop

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu

#SBATCH -c 4

#BATCH --mem=90G

#SBATCH -t 10:00:00

#SBATCH --nice=10000


source $HOME/.bashrc

echo Launching the training script

conda activate imCPA

python -config_file /home/icb/alessandro.palma/imCPA/configs/config_vae.json -mode train


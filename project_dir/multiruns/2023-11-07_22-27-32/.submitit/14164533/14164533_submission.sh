#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=6
#SBATCH --error=/home/icb/alessandro.palma/environment/IMPA/IMPA/IMPA/../project_dir/multiruns/2023-11-07_22-27-32/.submitit/%j/%j_0_log.err
#SBATCH --exclude=supergpu03,supergpu02
#SBATCH --gres=gpu:1
#SBATCH --job-name=main_hydra
#SBATCH --mem=90GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/home/icb/alessandro.palma/environment/IMPA/IMPA/IMPA/../project_dir/multiruns/2023-11-07_22-27-32/.submitit/%j/%j_0_log.out
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --signal=USR2@120
#SBATCH --time=1440
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/icb/alessandro.palma/environment/IMPA/IMPA/IMPA/../project_dir/multiruns/2023-11-07_22-27-32/.submitit/%j/%j_%t_log.out --error /home/icb/alessandro.palma/environment/IMPA/IMPA/IMPA/../project_dir/multiruns/2023-11-07_22-27-32/.submitit/%j/%j_%t_log.err /home/icb/alessandro.palma/miniconda3/envs/IMPA_try/bin/python -u -m submitit.core._submit /home/icb/alessandro.palma/environment/IMPA/IMPA/IMPA/../project_dir/multiruns/2023-11-07_22-27-32/.submitit/%j

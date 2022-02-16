#!/bin/bash

#SBATCH -o slurm_output3.txt
#SBATCH -e slurm_error3.txt
#SBATCH -J download
#SBATCH -p cpu_p
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH --nice=10000 

echo Collecting datasets
source get_data.sh

#!/bin/bash

#SBATCH -o data_loading.txt
#SBATCH -e data_loading.txt
#SBATCH -J DataLoadingAlessandro
#SBATCH -p cpu_p # in case you want to exclude individual nodes from your job submission
#SBATCH -c 1
#SBATCH --nice=10000

echo Getting the data 

source get_data.sh

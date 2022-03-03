#!/bin/bash

#SBATCH -o data_resize.txt
#SBATCH -e data__resize.txt
#SBATCH -J resize
#SBATCH -p gpu_p
#SBATCH -q gpu
#SBATCH -c 1
#SBATCH --mem=20
#SBATCH --nice=10000 

echo start resizing 

source $HOME/.bashrc

conda activate imgCPA

python3 ../resize_ds.py --data_dir /storage/groups/ml01/workspace/alessandro.palma/cellpainting_untar2 --out_dir /storage/groups/ml01/workspace/alessandro.palma/cellpainting_128 --width 128 --height 128 --interp cubic

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

cd /home/icb/alessandro.palma/imCPA

python3 -m compert.data.resize_ds --data_dir /storage/groups/ml01/workspace/alessandro.palma/cellpainting_512_dmso --out_dir /storage/groups/ml01/workspace/alessandro.palma/cellpainting_100_dmso --width 100 --height 100 --interp cubic

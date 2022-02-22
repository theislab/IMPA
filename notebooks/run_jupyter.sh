#!/bin/bash

source $HOME/.bashrc
echo 'Starting jupyter'
chmod 600 $HOME/data/slurm_jupyter_$SLURM_JOB_ID.job

# do stuff
cd 
conda activate imgCPA

# launch the jupyter instance (this works with jupyter notebook as well) 
jupyter-lab --no-browser --ip=0.0.0.0

rm $HOME/slurm_jupyter_$SLURM_JOB_ID.job


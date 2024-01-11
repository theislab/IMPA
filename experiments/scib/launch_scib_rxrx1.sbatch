#!/bin/bash

#SBATCH -o ./logs/scib_rxrx1.out

#SBATCH -e ./logs/scib_rxrx1.err

#SBATCH -J scib_rxrx1

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000


python compute_scib.py --dataset=rxrx1
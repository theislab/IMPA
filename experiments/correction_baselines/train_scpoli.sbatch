#!/bin/bash

#SBATCH -o ./logs/train_scpoli.out

#SBATCH -e ./logs/train_scpoli.err

#SBATCH -J train_scpoli

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA_try

python train_baseline.py --model_type scpoli --dataset_type rxrx1 --path_before_correction /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/featurized_anndata/rxrx1/rxrx1_adata_before_transf.h5ad --batch_key batch --data_index_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/datasets/rxrx1/metadata/rxrx1_df.csv --save_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/featurized_anndata/rxrx1/rxrx1_adata_scpoli.h5ad 
# python train_baseline.py --model_type scpoli --dataset_type cpg0000 --path_before_correction /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/featurized_anndata/cpg0000/cpg0000_adata_before_transf.h5ad --batch_key plate --data_index_path /lustre/groups/ml01/datasets/projects/cpg0000_alessandro/metadata/metadata_large.csv --save_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/featurized_anndata/cpg0000/cpg0000_adata_scpoli.h5ad 
# python train_baseline.py --model_type scpoli --dataset_type cpg0000 --path_before_correction /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/featurized_anndata/cpg0000/cpg0000_adata_before_transf_2.h5ad --batch_key plate --data_index_path /lustre/groups/ml01/datasets/projects/cpg0000_alessandro/metadata/metadata_large.csv --save_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/featurized_anndata/cpg0000/cpg0000_adata_scpoli_2.h5ad 

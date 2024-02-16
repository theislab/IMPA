#!/bin/bash

#SBATCH -o ./logs/train_scgen_2.out

#SBATCH -e ./logs/train_scgen_2.err

#SBATCH -J train_scgen_2

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate IMPA_try

# python compute_scib.py --scgen_anndata_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/featurized_anndata/cpg0000/cpg0000_adata_scgen.h5ad --scpoli_anndata_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/featurized_anndata/cpg0000/cpg0000_adata_scpoli.h5ad --impa_anndata_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/featurized_anndata/cpg0000/cpg0000_adata_after_transf.h5ad --unintegrated_adata /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/featurized_anndata/cpg0000/cpg0000_adata_before_transf.h5ad --dataset cpg0000 --scib_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/scib/cpg0000.pkl
python compute_scib.py --scgen_anndata_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/featurized_anndata/cpg0000/cpg0000_adata_scgen_2.h5ad --scpoli_anndata_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/featurized_anndata/cpg0000/cpg0000_adata_scpoli_2.h5ad --impa_anndata_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/featurized_anndata/cpg0000/cpg0000_adata_after_transf_2.h5ad --unintegrated_adata /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/featurized_anndata/cpg0000/cpg0000_adata_before_transf_2.h5ad --dataset cpg0000 --scib_path /home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/scib/cpg0000_2.pkl
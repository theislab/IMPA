import yaml
import torch
from IMPA.dataset.data_loader import CellDataLoader
from IMPA.solver import IMPAmodule
import pandas as pd
from omegaconf import OmegaConf
import scanpy as sc

import argparse
import numpy as np
import IMPA.featurizer.vision_transformer as vits
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scib_metrics.benchmark import Benchmarker

import pickle as pkl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute scib metrics.")
    parser.add_argument("--dataset", type=str, help="Choose the type of dataset between cpg0000 and rxrx1")

    args = parser.parse_args()
    dataset = args.dataset
    print(f"Using dataset {dataset}")
    
    feature_dest_folder = Path("/home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/featurized_anndata")

    if dataset == "cpg0000":
        adata_before_transf = sc.read_h5ad(feature_dest_folder / "cpg0000_adata_before_transf.h5ad")
        adata_after_transf = sc.read_h5ad(feature_dest_folder / "cpg0000_adata_after_transf.h5ad")
        data_index = pd.read_csv('/lustre/groups/ml01/datasets/projects/cpg0000_alessandro/metadata/metadata_large.csv', index_col=1)
        compound_names = []

        for row in adata_before_transf.obs.iterrows():
            plate = row[1].plate
            well = row[1].well
            view = row[1]["view"]
            no = row[1].no
            file_name = f"{plate}_{well}_{view}_{no}"
            cpd = data_index.loc[file_name].CPD_NAME
            compound_names.append(cpd)
            
        adata_before_transf.obs["compound"] = compound_names
        adata_after_transf.obs["compound"] = compound_names
        adata_unique = adata_before_transf.copy()
        adata_unique.obsm["Unintegrated"] = adata_before_transf.obsm["X_pca"]
        adata_unique.obsm["IMPA-Integrated"] = adata_after_transf.obsm["X_pca"]
        bm = Benchmarker(
            adata_unique[:200000],
            batch_key="plate",
            label_key="compound",
            embedding_obsm_keys=["Unintegrated", "IMPA-Integrated"],
            n_jobs=4)
        bm.benchmark()
        scib_path = "/home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/scib/cpg0000.pkl"
    
    else:
        adata_before_transf = sc.read_h5ad(feature_dest_folder / "rxrx1_adata_before_transf.h5ad")
        adata_after_transf = sc.read_h5ad(feature_dest_folder / "rxrx1_adata_after_transf.h5ad")    
        data_index = pd.read_csv('/home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/datasets/rxrx1/metadata/rxrx1_df.csv', index_col=1)
        compound_names = []

        for row in adata_before_transf.obs.iterrows():
            batch = row[1].batch
            plate = row[1].plate
            well = row[1].well
            view = row[1]["view"]
            no = row[1].no
            file_name = f"U2OS-{batch}_{plate}_{well}_{view}_{no}"
            cpd = data_index.loc[file_name].CPD_NAME
            compound_names.append(cpd)
            
        adata_before_transf.obs["compound"] = compound_names
        adata_after_transf.obs["compound"] = compound_names
        adata_unique = adata_before_transf.copy()
        adata_unique.obsm["Unintegrated"] = adata_before_transf.obsm["X_pca"]
        adata_unique.obsm["IMPA-Integrated"] = adata_after_transf.obsm["X_pca"]
        
        bm = Benchmarker(
                adata_unique,
                batch_key="batch",
                label_key="compound",
                embedding_obsm_keys=["Unintegrated", "IMPA-Integrated"],
                n_jobs=4,
            )
        bm.benchmark()
        scib_path = "/home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/dino_featurization_project/scib/rxrx1.pkl"

    with open(scib_path, "wb") as file:
        pkl.dump(bm, file)
    
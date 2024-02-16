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
from scib_metrics.benchmark import BioConservation, BatchCorrection

import pickle as pkl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute scib metrics.")
    # Read anndata
    parser.add_argument("--scgen_anndata_path")
    parser.add_argument("--scpoli_anndata_path")
    parser.add_argument("--impa_anndata_path")
    parser.add_argument("--unintegrated_adata")
    parser.add_argument("--dataset")
    parser.add_argument("--scib_path")

    args = parser.parse_args()
    scgen_anndata_path = args.scgen_anndata_path
    scpoli_anndata_path = args.scpoli_anndata_path
    impa_anndata_path = args.impa_anndata_path
    unintegrated_adata = args.unintegrated_adata
    scib_path = args.scib_path
    dataset = args.dataset

    scgen_anndata = sc.read_h5ad(scgen_anndata_path)
    scpoli_anndata = sc.read_h5ad(scpoli_anndata_path)
    impa_anndata = sc.read_h5ad(impa_anndata_path)
    unintegrated_adata = sc.read_h5ad(unintegrated_adata)    
    
    adata_unique = scgen_anndata.copy()
    adata_unique.obsm["X_scgen"] = scgen_anndata.X.copy()
    adata_unique.obsm["X_scipoli"] = scgen_anndata.X.copy()
    adata_unique.obsm["X_impa"] = impa_anndata.X.copy()
    adata_unique.obsm["X_unintegrated"] = unintegrated_adata.X.copy()
    
    biocons = BioConservation(isolated_labels=False, 
                              nmi_ari_cluster_labels_leiden=False, 
                              nmi_ari_cluster_labels_kmeans=False,
                              silhouette_label=True,
                              clisi_knn=False)
    
    batch_correction = BatchCorrection(silhouette_batch=True,
                              ilisi_knn=True, 
                              kbet_per_label=True, 
                              graph_connectivity=True, 
                              pcr_comparison=True)

    if dataset == "cpg0000":
        bm = Benchmarker(
            adata_unique[:50000],
            batch_key="plate",
            label_key="compound",
            bio_conservation_metrics=biocons,
            batch_correction_metrics=batch_correction,
            embedding_obsm_keys=["X_scgen", "X_scipoli", "X_unintegrated", "X_impa" ],
            n_jobs=4)
        bm.benchmark()
    
    else:
        bm = Benchmarker(
                adata_unique[:50000],
                batch_key="batch",
                label_key="compound",
                bio_conservation_metrics=biocons,
                batch_correction_metrics=batch_correction,
                embedding_obsm_keys=["X_scgen", "X_scipoli", "X_unintegrated", "X_impa"],
                n_jobs=4,
            )
        bm.benchmark()

    with open(scib_path, "wb") as file:
        pkl.dump(bm, file)
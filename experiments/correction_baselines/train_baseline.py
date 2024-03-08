import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import scib
import seaborn as sns
import pertpy as pt
import warnings
from matplotlib import rcParams
from scarches.models.scpoli import scPoli
import argparse

import warnings
from matplotlib import rcParams

def main(args):
    # Assign variables
    model_type = args.model_type
    dataset_type = args.dataset_type 
    path_before_correction = args.path_before_correction
    batch_key = args.batch_key
    data_index_path = args.data_index_path
    save_path = args.save_path
    
    # Read the data before correction 
    adata_before_correction = sc.read_h5ad(path_before_correction)
    
    # Assign compound name
    data_index = pd.read_csv(data_index_path, index_col=1)

    compound_names = []
    if dataset_type == "rxrx1":
        for row in adata_before_correction.obs.iterrows():
            batch = row[1].batch
            plate = row[1].plate
            well = row[1].well
            view = row[1]["view"]
            no = row[1].no
            file_name = f"U2OS-{batch}_{plate}_{well}_{view}_{no}"
            cpd = data_index.loc[file_name].CPD_NAME
            compound_names.append(cpd)
    
    else:
        for row in adata_before_correction.obs.iterrows():
            plate = row[1].plate
            well = row[1].well
            view = row[1]["view"]
            no = row[1].no
            file_name = f"{plate}_{well}_{view}_{no}"
            cpd = data_index.loc[file_name].CPD_NAME
            compound_names.append(cpd)
    
    # Add compound key
    adata_before_correction.obs["compound"] = compound_names
    
    # Assign hyperparameters for early stopping
    if model_type == "scpoli":
        early_stopping_kwargs = {"early_stopping_metric": "val_prototype_loss",
                                    "mode": "min",
                                    "threshold": 0,
                                    "patience": 20,
                                    "reduce_lr": True,
                                    "lr_patience": 13,
                                    "lr_factor": 0.1}
        
        scpoli_model = scPoli(
            adata=adata_before_correction,
            condition_keys=batch_key,
            cell_type_keys='compound',
            embedding_dims=30,
            recon_loss='mse',
            latent_dim=30
        )
        scpoli_model.train(
            n_epochs=100,
            pretraining_epochs=10,
            early_stopping_kwargs=early_stopping_kwargs,
            eta=5,
        )
        
        scpoli_model.model.eval()
        data_latent = scpoli_model.get_latent(adata_before_correction,
                                                mean=True)
        adata_latent = sc.AnnData(data_latent)
        adata_latent.obs = adata_before_correction.obs.copy()
        
    elif model_type=="scgen":
        pt.tl.SCGEN.setup_anndata(adata_before_correction, batch_key=batch_key, labels_key='compound')
        model = pt.tl.SCGEN(adata_before_correction)
        model.train(
            max_epochs=100,
            batch_size=32,
            early_stopping=True,
            early_stopping_patience=25, 
            accelerator="gpu")  
        adata_before_correction_with_latent = model.batch_removal()
        adata_latent = sc.AnnData(adata_before_correction_with_latent.obsm["corrected_latent"])
        adata_latent.obs = adata_before_correction.obs.copy()
    
    elif model_type=="harmony":
        scib.ig.harmony(adata_before_correction, batch="batch")
        
    elif model_type=="scanorama":
        scib.ig.scanorama(adata_before_correction, batch="batch")
        
    elif model_type=="combat":
        scib.ig.combat(adata_before_correction, batch="batch")
        
    elif model_type=="mpnn":
        scib.ig.mpnn(adata_before_correction_with_latent, batch="batch")
        
    if model_type in ["scgen", "scpoli"]:    
        sc.pp.pca(adata_latent)
        sc.pp.neighbors(adata_latent)
        sc.tl.umap(adata_latent)
        adata_latent.write_h5ad(save_path)
    else:
        adata_before_correction.X = adata_before_correction.obsm["X_emb"]
        adata_before_correction.write_h5ad(save_path)
    
    
if __name__ == "__main__":
    # Set up the the arguments 
    parser = argparse.ArgumentParser(description="Add the input information")
    # Add arguments 
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--path_before_correction", type=str)
    parser.add_argument("--batch_key", type=str)
    parser.add_argument("--data_index_path", type=str)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    
    # Start main
    main(args)

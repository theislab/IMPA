import sys

sys.path.insert(0, '../')

from collections import OrderedDict
from tqdm import tqdm
from os.path import join as ospj

import ot
import numpy as np
import torch
import pickle as pkl

# from compert.core.utils import save_image 
from core.utils import save_image, swap_attributes

from .gan_metrics.fid import *

def calculate_rmse_and_disentanglement_score(nets, 
                                            loader, 
                                            device, 
                                            dest_dir,
                                            embedding_path, 
                                            args, 
                                            embedding_matrix):

    # The difference between a decoded image with and without addition of the drug and moa
    rmse_transformations = 0  

    # The lists containing the true labels of the batch 
    y_true_ds = []  
    y_fake_ds = []
    X_real = []
    X_swapped = []

    # STORE THE LATENTS FOR LATER ANALYSIS
    z_basal_ds = []  
    z_style_ds = []

    # LOOP OVER SINGLE OBSERVATIONS 
    for observation in tqdm(loader):
        # Data matrix
        X = observation['X'].to(device)  
        # Labelled observations
        y_one_hot_src = observation['mol_one_hot'].to(device) 
        y_src = y_one_hot_src.argmax(1).long()
        y_trg = swap_attributes(y_one_hot_src, y_src, device).long().argmax(1)

        # Append drug label to cache
        y_true_ds.append(y_src.to('cpu'))  # Record the labels 
        y_fake_ds.append(y_trg.to('cpu'))

        # Prepare noise if the stochastic version is run 
        if args.stochastic > 0:
            # Draw random vector 
            z = torch.randn(X.shape[0], args.z_dimension).to(device)

        with torch.no_grad():
            # We generate in normal mode
            z_emb = embedding_matrix(y_trg)

            if args.stochastic:
                z_emb = torch.cat([z_emb, z], dim=1)
            
            # Map to style
            s = nets.mapping_network(z_emb) 
            z_basal, X_fake = nets.generator(X, s)
            X_swapped.append(X_fake.cpu())
            X_real.append(X.cpu())
            
            # Save the basal state and style for later visualization 
            z_basal_ds.append(z_basal.detach().to('cpu'))
            z_style_ds.append(s.detach().to('cpu'))


    # Perform list concatenation on all of the results 
    y_true_ds = torch.cat(y_true_ds).to('cpu').numpy()
    y_fake_ds = torch.cat(y_fake_ds).to('cpu').numpy()
    X_swapped = torch.cat(X_swapped, dim=0)
    X_swapped = X_swapped.view(len(X_swapped),-1)
    X_real = torch.cat(X_real, dim=0)
    X_real = X_real.view(len(X_real),-1)
    categories = np.unique(y_true_ds)

    # Update the RMSEs
    for cat in tqdm(categories):
        X_real_cat = X_real[y_true_ds==cat]
        X_swapped_cat = X_swapped[y_fake_ds==cat]
        diff = wd_score=ot.emd2(torch.tensor([]), torch.tensor([]), ot.dist(X_real_cat, 
                                                                                X_swapped_cat, 
                                                                                'euclidean'), 1)
        # diff =  np.mean(np.sqrt(np.mean((X_real_cat - X_swapped_cat)**2, axis=2)))
        rmse_transformations += diff

    # Transform basal state and labels to tensors
    z_basal_ds = torch.cat(z_basal_ds, dim=0).detach().cpu().numpy()
    z_style_ds = torch.cat(z_style_ds, dim=0).cpu().numpy()

    # Print metrics 
    dict_metrics = {'rmse_transformations': rmse_transformations/len(categories)}

    # Embeddings 
    emb_path = ospj(dest_dir, embedding_path, 'embeddings.pkl')
    print(f"Save embeddings at {emb_path}")
    with open(emb_path, 'wb') as file:
        pkl.dump([z_basal_ds, z_style_ds, y_fake_ds], file)

    return dict_metrics

import pickle as pkl

import numpy as np
import torch
from os.path import join as ospj

import ot
from .gan_metrics.fid import *
from tqdm import tqdm
from utils import swap_attributes


def evaluate(nets, 
            loader, 
            device, 
            dest_dir,
            embedding_path, 
            args, 
            embedding_matrix, 
            channels=[0,1,2]):
    
    """Evaluate the model during training

    Args:
        nets (dict): dictionary with model networks
        loader (torch.data.utils.DataLoader): data loading object
        device (str): `cuda` or `cpu`
        dest_dir (str): directory where to save embeddings 
        embedding_path (str): path to the destination directory
        args (dict): hyperparameters for the training 
        embedding_matrix (torch.Tensor): tensor with perturbation embeddings

    Returns:
        dict: a dictionary storing the evaluation metrics
    """
    
    # Accumulated distance between true and generated images 
    wd_transformations = 0  
    fid_transformations = 0  

    # The lists containing the true labels of the batch 
    y_true_ds = []  
    y_fake_ds = []
    X_real = []
    X_swapped = []

    # Store points for the style and the content encoding 
    z_basal_ds = []  
    z_style_ds = []

    # Loop over single observations
    for observation in tqdm(loader):
        # Get the data and swap the labels 
        X = observation['X'].to(device)  
        y_one_hot_org = observation['mol_one_hot'].to(device) 
        y_org = y_one_hot_org.argmax(1).long()
        y_trg = swap_attributes(y_one_hot_org, y_org, device).long().argmax(1) # random swap

        # Store perturbation labels
        y_true_ds.append(y_org.to('cpu'))  
        y_fake_ds.append(y_trg.to('cpu'))

        # Draw random vector for style conditioning
        if args.stochastic:
            z = torch.randn(X.shape[0], args.z_dimension).to(device)

        with torch.no_grad():
            # Get pertrubation embedding and concatenate with the noise vector 
            z_emb = embedding_matrix(y_trg)
            if args.stochastic:
                z_emb = torch.cat([z_emb, z], dim=1)
            
            # Map to style
            s = nets.mapping_network(z_emb) 
            
            # Generate
            z_basal, X_fake = nets.generator(X, s)
            
            # Saved real and swapped images 
            X_swapped.append(X_fake.cpu())
            X_real.append(X.cpu())
            
            # Save the basal state and style for later visualization 
            z_basal_ds.append(z_basal.detach().to('cpu'))
            z_style_ds.append(s.detach().to('cpu'))


    # Perform list concatenation on all of the results 
    y_true_ds = torch.cat(y_true_ds).to('cpu').numpy()
    y_fake_ds = torch.cat(y_fake_ds).to('cpu').numpy()
    
    # Concatenate and get the flattened versions of the images
    X_swapped = torch.cat(X_swapped, dim=0)
    X_real = torch.cat(X_real, dim=0)
    categories = np.unique(y_true_ds)
    
    if len(categories) > 100:
        categories = np.random.choice(categories, 100)

    # Update the metrics scores (FID, WD)
    for cat in tqdm(categories):
        # Compute FID/WD for a class at a time 
        X_real_cat = X_real[y_true_ds==cat]
        X_swapped_cat = X_swapped[y_fake_ds==cat]
        
        # Wasserstein distance
        wd = ot.emd2(torch.tensor([]), torch.tensor([]), ot.dist(X_real_cat.view(len(X_real_cat),-1), 
                                                                                X_swapped_cat.view(len(X_swapped_cat),-1), 
                                                                                'euclidean'), 1)
        wd_transformations += wd
        
        # FID  
        X_real_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_real_cat.to('cuda')))
        X_swapped_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_swapped_cat.to('cuda')))
        fid = cal_fid(X_real_dataset, X_swapped_dataset, 2048, True, custom_channels=channels)
        fid_transformations += fid

    # Concatenate style and content vectors 
    z_basal_ds = torch.cat(z_basal_ds, dim=0).detach().cpu().numpy()
    z_style_ds = torch.cat(z_style_ds, dim=0).detach().cpu().numpy()

    # Save metrics 
    dict_metrics = {'wd_transformations': wd_transformations/len(categories), 
                    'fid_transformations': fid_transformations/len(categories)}

    # Dump latent embeddings  
    emb_path = ospj(dest_dir, embedding_path, 'embeddings.pkl')
    print(f"Save embeddings at {emb_path}")
    with open(emb_path, 'wb') as file:
        pkl.dump([z_basal_ds, z_style_ds, y_fake_ds], file)

    return dict_metrics

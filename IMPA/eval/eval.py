import pickle as pkl
import numpy as np
import random
import torch
from os.path import join as ospj

import ot
from IMPA.eval.gan_metrics.fid import *
from tqdm import tqdm
from IMPA.utils import swap_attributes
import torch.nn.functional as F


def evaluate(nets, loader, device, args, embedding_matrix, batch_correction, n_classes, channels=[0,1,2]):
    """Evaluate the model during training.

    Args:
        nets (dict): Dictionary with model networks.
        loader (torch.utils.data.DataLoader): Data loading object.
        device (str): 'cuda' or 'cpu'.
        args (dict): Hyperparameters for training.
        embedding_matrix (torch.Tensor): Tensor with perturbation embeddings.
        batch_correction (bool): Whether batch correction is applied.
        n_classes (int): Number of classes.
        channels (list, optional): List of channel indices for evaluation. Defaults to [0, 1, 2].

    Returns:
        dict: A dictionary storing the evaluation metrics.
    """

    # Initialize accumulated metrics
    if not batch_correction:
        wd_transformations = 0
    fid_transformations = 0

    # Lists to store the true labels and generated images
    X_real = []
    X_pred = []
    Y_trg = []
    if batch_correction:
        Y_org = []

    # Loop over single observations
    for observation in tqdm(loader):
        if batch_correction:
            x_real = observation['X'].to(device)
            y_org = observation['mols'].long().to(device)
            y_trg = swap_attributes(n_classes, y_org, device)
            Y_org.append(y_org.to('cpu'))
        else:
            x_real_ctrl, x_real_trt = observation['X']
            x_real_ctrl, x_real_trt = x_real_ctrl.to(device), x_real_trt.to(device)
            y_trg = observation['mols'].long().to(device)
    
        # Store perturbation labels
        Y_trg.append(y_trg.to('cpu'))
 
        # Draw random vector for style conditioning
        if args.stochastic:
            if batch_correction:
                z = torch.randn(x_real.shape[0], args.z_dimension).to(device)
            else:
                z = torch.randn(x_real_ctrl.shape[0], args.z_dimension).to(device)

        with torch.no_grad():
            # Get perturbation embedding and concatenate with the noise vector
            z_emb = embedding_matrix(y_trg)
            z_emb = torch.cat([z_emb, z], dim=1)

            # Map to style
            s = nets.mapping_network(z_emb)

            # Generate images
            if batch_correction:
                _, x_fake = nets.generator(x_real, s)
            else:
                _, x_fake = nets.generator(x_real_ctrl, s)
                
            X_pred.append(x_fake.cpu())
            if batch_correction:
                X_real.append(x_real.cpu())  
            else:
                X_real.append(x_real_trt.cpu())  

    # Perform list concatenation on all of the results
    if batch_correction:
        Y_org = torch.cat(Y_org).to('cpu').numpy()
    Y_trg = torch.cat(Y_trg).to('cpu').numpy()
    categories = np.unique(Y_trg)

    # Concatenate and get the flattened versions of the images
    X_pred = torch.cat(X_pred, dim=0)
    X_real = torch.cat(X_real, dim=0)
    
    # Evaluate on 100  conditions at random if there are more than 100 conditions
    if len(categories) > 100:
        categories = np.random.choice(categories, 100)

    # Update the metrics scores (FID, WD)
    for cat in tqdm(categories):
        # Compute FID/WD for a class at a time
        if batch_correction:
            X_real_cat = X_real[Y_org == cat]
        else:
            X_real_cat = X_real[Y_trg == cat]  # Real cells from a category
        X_pred_cat = X_pred[Y_trg == cat]

        if not batch_correction:
            # Wasserstein distance
            wd = ot.emd2(torch.tensor([]), torch.tensor([]),
                        ot.dist(X_real_cat.view(len(X_real_cat), -1),
                                X_pred_cat.view(len(X_pred_cat), -1), 'euclidean'), 1)
            wd_transformations += wd

        else:
            num_samples_to_keep =  200
            indices_to_keep_real = random.sample(range(len(X_real_cat)), num_samples_to_keep)
            indices_to_keep_generated = random.sample(range(len(X_pred_cat)), num_samples_to_keep)

            X_real_cat = X_real_cat[indices_to_keep_real]
            X_pred_cat = X_pred_cat[indices_to_keep_generated]
            
        # FID
        X_real_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_real_cat.to(device)))
        X_pred_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_pred_cat.to(device)))
        fid = cal_fid(X_real_dataset, X_pred_dataset, 2048, True, custom_channels=channels)
        fid_transformations += fid

    # Save metrics
    if batch_correction:
        dict_metrics = {'fid_transformations': fid_transformations / len(categories)}
    else:        
        dict_metrics = {'wd_transformations': wd_transformations / len(categories),
                        'fid_transformations': fid_transformations / len(categories)}

    return dict_metrics

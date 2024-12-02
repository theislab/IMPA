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
    Y_trg = []  # Target condition
    X_real = []  # Real images 
    X_pred = []  # Predicted images
    
    if batch_correction:
        Y_org = []
    if args.multimodal:
        Y_mod = []

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
            y_trg = observation['mols'].long().to(device)  # The modality-specific category of interest 
            y_mod = observation['y_id'].long().to(device)  # The perturbation modality of an observation
        
        # Draw random vector for style conditioning
        if args.stochastic:
            if batch_correction:
                z = torch.randn(x_real.shape[0], args.z_dimension).to(device)
            else:
                z = torch.randn(x_real_ctrl.shape[0], args.z_dimension).to(device)

        with torch.no_grad():
            # Get perturbation embedding and concatenate with the noise vector
            if not args.multimodal:
                z_emb = embedding_matrix(y_trg)
                if args.stochastic:
                    z_emb = torch.cat([z_emb, z], dim=1)
                s = nets.mapping_network(z_emb)
            else:
                # If model is multimodal
                x_real_ctrl_reordered = []
                x_real_trt_reordered = []
                y_trg_reorderd = []
                y_mod_reordered = []
                z_emb = []
                s = []                
                for i in range(args.n_mod):
                    index_mod = (y_mod==i)
                    x_real_ctrl_mod = x_real_ctrl[index_mod]
                    x_real_trt_mod = x_real_trt[index_mod]
                    y_trg_mod = y_trg[index_mod] 
                    y_mod_mod = y_mod[index_mod]
                    
                    z_emb_mod = embedding_matrix[i](y_trg_mod)
                    if args.stochastic:
                        z_emb_mod = torch.cat([z_emb_mod, z[index_mod]], dim=1)
                    
                    x_real_ctrl_reordered.append(x_real_ctrl_mod)
                    x_real_trt_reordered.append(x_real_trt_mod)
                    y_trg_reorderd.append(y_trg_mod)
                    y_mod_reordered.append(y_mod_mod)
                    z_emb.append(z_emb_mod)
                    s.append(nets.mapping_network(z_emb_mod, y_trg_mod, i))
                
                x_real_ctrl = torch.cat(x_real_ctrl_reordered, dim=0)   
                x_real_trt = torch.cat(x_real_trt_reordered, dim=0)   
                y_trg = torch.cat(y_trg_reorderd, dim=0)   
                y_mod = torch.cat(y_mod_reordered, dim=0) 
                del x_real_ctrl_reordered
                del x_real_trt_reordered
                del y_trg_reorderd
                del y_mod_reordered
                z_emb = torch.cat(z_emb, dim=0)   
                s = torch.cat(s, dim=0)            

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
            
            # Store perturbation labels
            Y_trg.append(y_trg.to('cpu'))
            if args.multimodal:
                Y_mod.append(y_mod.to('cpu'))

    # Concatenate batches
    if batch_correction:
        Y_org = torch.cat(Y_org).to('cpu').numpy()
    # Concatenate perturbation index 
    if args.multimodal:
        Y_mod = torch.cat(Y_mod).to('cpu').numpy()
    # Concatenate the perturbation id
    Y_trg = torch.cat(Y_trg).to('cpu').numpy()
    
    if not args.multimodal:
        categories = np.unique(Y_trg)
    else:
        categories = []
        for i in range(args.n_mod):
            idx_mod = (Y_mod==i)
            categories.append(np.unique(Y_trg[idx_mod]))

    # Concatenate and get the flattened versions of the images
    X_pred = torch.cat(X_pred, dim=0)
    X_real = torch.cat(X_real, dim=0)
    
    # Evaluate on 100  conditions at random if there are more than 100 conditions
    if not args.multimodal:
        if len(categories) > 100:
            categories = np.random.choice(categories, 100)
    else:
        categories = [np.random.choice(cat, 100) for cat in categories]

    # Update the metrics scores (FID, WD)
    if not args.multimodal:
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
        
    else:
        for i, cat in enumerate(categories):  # Different categories drawn per modality 
            for cat_mod in tqdm(cat):  #  For each of the categories in the selected modality
                X_real_cat = X_real[Y_mod == i][Y_trg == cat_mod]  # Real cells from a category
                X_pred_cat = X_pred[Y_mod == i][Y_trg == cat_mod]
                
                num_samples_to_keep =  min(200, len(X_real_cat))
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

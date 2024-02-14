import pickle as pkl

import random
import numpy as np
import torch
from os.path import join as ospj

import ot
from IMPA.eval.gan_metrics.fid import *
from tqdm import tqdm
from IMPA.utils import swap_attributes


def evaluate(nets, loader, device, dest_dir, embedding_path, args, embedding_matrix, channels=[0,1,2], subsample_frac=1):
    """Evaluate the model during training.

    Args:
        nets (dict): Dictionary with model networks.
        loader (torch.utils.data.DataLoader): Data loading object.
        device (str): 'cuda' or 'cpu'.
        dest_dir (str): Directory where to save embeddings.
        embedding_path (str): Path to the destination directory.
        args (dict): Hyperparameters for training.
        embedding_matrix (torch.Tensor): Tensor with perturbation embeddings.
        channels (list, optional): List of channel indices for evaluation. Defaults to [0, 1, 2].

    Returns:
        dict: A dictionary storing the evaluation metrics.
    """

    # Accumulated distance between true and generated images
    fid_transformations = 0

    # Lists containing the true labels of the batch
    y_true_ds = []
    y_fake_ds = []
    X_real = []
    X_swapped = []

    # Loop over single observations
    for observation in tqdm(loader):
        # Get the data and swap the labels
        X = observation['X'].to(device)
        y_one_hot_org = observation['mol_one_hot'].to(device)
        y_org = y_one_hot_org.argmax(1).long()
        y_trg = swap_attributes(y_one_hot_org, y_org, device).long().argmax(1)  # random swap

        # Store perturbation labels
        y_true_ds.append(y_org.to('cpu'))
        y_fake_ds.append(y_trg.to('cpu'))

        # Draw random vector for style conditioning
        if args.stochastic:
            z = torch.randn(X.shape[0], args.z_dimension).to(device)

        with torch.no_grad():
            # Get perturbation embedding and concatenate with the noise vector
            z_emb = embedding_matrix(y_trg)
            if args.stochastic:
                z_emb = torch.cat([z_emb, z], dim=1)

            # Map to style
            s = nets.mapping_network(z_emb)

            # Generate
            _, X_fake = nets.generator(X, s)

            # Save real and swapped images
            X_swapped.append(X_fake.cpu())
            X_real.append(X.cpu())

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
        X_real_cat = X_real[y_true_ds == cat]
        X_swapped_cat = X_swapped[y_fake_ds == cat]
        
        # Randomly subsample data
        # num_samples_to_keep_real = int(len(X_real_cat) * subsample_frac
        num_samples_to_keep_real =  100
        indices_to_keep_real = random.sample(range(len(X_real_cat)), num_samples_to_keep_real)
        # num_samples_to_keep_generated = int(len(X_swapped_cat) * subsample_frac)
        num_samples_to_keep_generated = 100
        indices_to_keep_generated = random.sample(range(len(X_swapped_cat)), num_samples_to_keep_generated)

        X_real_cat = X_real_cat[indices_to_keep_real]
        X_swapped_cat = X_swapped_cat[indices_to_keep_generated]

        # FID
        print("FID")
        X_real_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_real_cat.to(device)))
        X_swapped_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_swapped_cat.to(device)))
        fid = cal_fid(X_real_dataset, X_swapped_dataset, 2048, True, custom_channels=channels)
        fid_transformations += fid
        
        del X_real_cat
        del X_swapped_cat

    del X_swapped
    del X_real
    
    # Save metrics
    dict_metrics = {'fid_transformations': fid_transformations / len(categories)}
    
    return dict_metrics

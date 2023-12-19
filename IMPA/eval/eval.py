import pickle as pkl

import numpy as np
import torch
from os.path import join as ospj

import ot
from IMPA.eval.gan_metrics.fid import *
from tqdm import tqdm
from IMPA.utils import swap_attributes


def evaluate(nets, loader, device, dest_dir, embedding_path, args, embedding_matrix, channels=[0,1,2]):
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
    wd_transformations = 0
    fid_transformations = 0

    # Lists containing the true labels of the batch
    Y_trg = []
    X_real = []
    X_pred = []

    # Store points for the style and content encoding
    Z_basal = []
    Z_style = []

    # Loop over single observations
    for observation in tqdm(loader):
        # Get the data and swap the labels
        x_real_ctrl, x_real_trt = observation['X']
        x_real_ctrl, x_real_trt = x_real_ctrl.to(device), x_real_trt.to(device)
        y_one_hot_trg = observation['mol_one_hot'].to(device)
        y_trg = y_one_hot_trg.argmax(1).long()

        # Store perturbation labels
        Y_trg.append(y_trg.to('cpu'))

        # Draw random vector for style conditioning
        if args.stochastic:
            z = torch.randn(x_real_ctrl.shape[0], args.z_dimension).to(device)

        with torch.no_grad():
            # Get perturbation embedding and concatenate with the noise vector
            z_emb = embedding_matrix(y_trg)
            if args.stochastic:
                z_emb = torch.cat([z_emb, z], dim=1)

            # Map to style
            s = nets.mapping_network(z_emb)

            # Generate
            z_basal, x_fake = nets.generator(x_real_ctrl, s)

            # Save real and swapped images
            X_pred.append(x_fake.cpu())
            X_real.append(x_real_trt.cpu())

            # Save the basal state and style for later visualization
            Z_basal.append(z_basal.detach().to('cpu'))
            Z_style.append(s.detach().to('cpu'))

    # Perform list concatenation on all of the results
    Y_trg = torch.cat(Y_trg).to('cpu').numpy()

    # Concatenate and get the flattened versions of the images
    X_pred = torch.cat(X_pred, dim=0)
    X_real = torch.cat(X_real, dim=0)
    categories = np.unique(Y_trg)

    if len(categories) > 100:
        categories = np.random.choice(categories, 100)

    # Update the metrics scores (FID, WD)
    for cat in tqdm(categories):
        # Compute FID/WD for a class at a time
        X_real_cat = X_real[Y_trg == cat]
        X_pred_cat = X_pred[Y_trg == cat]

        # Wasserstein distance
        wd = ot.emd2(torch.tensor([]), torch.tensor([]),
                     ot.dist(X_real_cat.view(len(X_real_cat), -1),
                             X_pred_cat.view(len(X_pred_cat), -1), 'euclidean'), 1)
        wd_transformations += wd

        # FID
        X_real_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_real_cat.to(device)), batch_size=64)
        X_pred_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_pred_cat.to(device)), batch_size=64)
        fid = cal_fid(X_real_dataset, X_pred_dataset, 2048, True, custom_channels=channels)
        fid_transformations += fid

    # Concatenate style and content vectors
    Z_basal = torch.cat(Z_basal, dim=0).detach().cpu().numpy()
    Z_style = torch.cat(Z_style, dim=0).detach().cpu().numpy()

    # Save metrics
    dict_metrics = {'wd_transformations': wd_transformations / len(categories),
                    'fid_transformations': wd_transformations / len(categories)}

    # Dump latent embeddings
    emb_path = ospj(dest_dir, embedding_path, 'embeddings.pkl')
    print(f"Save embeddings at {emb_path}")
    with open(emb_path, 'wb') as file:
        pkl.dump([Z_basal, Z_style, Y_trg], file)

    return dict_metrics

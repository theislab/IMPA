import sys

sys.path.insert(0, '../')

from collections import OrderedDict
from tqdm import tqdm
from os.path import join as ospj

import numpy as np
import torch
import pickle as pkl

# from compert.core.utils import save_image 
from core.utils import save_image 

from .disentanglement_score import compute_disentanglement_score

def calculate_rmse_and_disentanglement_score(nets, 
                                            loader, 
                                            device, 
                                            dest_dir,
                                            embedding_path, 
                                            args, 
                                            step,
                                            embedding_matrix):

    # The difference between a decoded image with and without addition of the drug and moa
    rmse_basal_full = 0  
    rmse = 0 

    # The lists containing the true labels of the batch 
    y_true_ds_drugs = []  

    # STORE THE LATENTS FOR LATER ANALYSIS
    z_basal_ds = []  
    z_style = []

    # IMAGES TO PLOT
    x_rec_post_to_plot = None
    x_rec_rand_to_plot = None
    x_rec_basal_to_plot = None

    # LOOP OVER SINGLE OBSERVATIONS 
    for observation in tqdm(loader):
        # Data matrix
        X = observation['X'].to(device)  
        # Labelled observations
        y_one_hot = observation['mol_one_hot'].to(device) 
        y = y_one_hot.argmax(1) 
        # Append drug label to cache
        y_true_ds_drugs.append(y.to('cpu'))  # Record the labels 

        # Prepare noise if the stochastic version is run 
        if args.stochastic > 0:
            # Draw random vector 
            z = torch.randn(X.shape[0], args.z_dimension).to(device)

        with torch.no_grad():
            # We generate in basal mode, so without really conditining on s_trg
            s = None
            z_basal, x_rec_basal = nets.generator(X, s, basal=True)  

            # We generate in normal mode
            z_emb = embedding_matrix(y)
            if args.stochastic:
                z_emb = torch.cat([z_emb, z], dim=1)

            s = nets.mapping_network(z_emb) if args.encode_rdkit else z_emb

            _, x_rec_rand = nets.generator(X, s)

            # Reconstruct from style vector - can be multi-task style encoding or single task style encoding 
            if not args.single_style:
                s_post = nets.style_encoder(X, y)
            else:
                s_post = nets.style_encoder(X)  
            _, x_rec_post = nets.generator(X, s_post)

            # Image that will be plotted 
            if x_rec_rand_to_plot == None and x_rec_basal_to_plot == None and args.dataset_name == 'bbbc021':
                x_rec_rand_to_plot = x_rec_rand
                x_rec_basal_to_plot = x_rec_basal
                x_rec_post_to_plot = x_rec_post
        
        # Update the RMSEs
        rmse_basal_full += torch.sqrt(torch.mean((x_rec_post-x_rec_basal)**2))
        rmse += torch.sqrt(torch.mean((x_rec_post-X)**2))

        # Save the basal state and style for later visualization 
        z_basal_ds.append(z_basal.detach().to('cpu'))
        z_style.append(s_post.detach().to('cpu'))

    del X
    del y_one_hot
    del z

    # Transform basal state and labels to tensors
    z_basal_ds = torch.cat(z_basal_ds, dim=0)
    z_style = torch.cat(z_style, dim=0)
    y_true_ds_drugs = torch.cat(y_true_ds_drugs, dim=0).to('cpu').numpy()
    # disentanglement_score = compute_disentanglement_score(z_basal_ds, y_true_ds_drugs)

    # Print metrics 
    dict_metrics = {'rmse': rmse.item()/len(loader),
                    'rmse_basal_full': rmse_basal_full.item()/len(loader)}
    
    # Plot the last reconstructed basal 
    filename_basal = ospj(dest_dir, args.basal_vs_real_folder, '%06d_basal_decoded_latent.png' % (step))
    filename_rec_rand = ospj(dest_dir, args.basal_vs_real_folder, '%06d_random_decoded_latent.png' % (step))
    filename_rec_post = ospj(dest_dir, args.basal_vs_real_folder, '%06d_post_decoded_latent.png' % (step))
    
    if args.dataset_name == 'bbbc021':
        save_image(x_rec_basal_to_plot[:16], 4, filename_basal)
        save_image(x_rec_post_to_plot[:16], 4, filename_rec_post)
        save_image(x_rec_rand_to_plot[:16], 4, filename_rec_rand)


    emb_path = ospj(dest_dir, embedding_path, 'embeddings.pkl')
    print(f"Save embeddings at {emb_path}")
    z_basal_ds = z_basal_ds.detach().cpu().numpy()
    z_style = z_style.detach().cpu().numpy()
    with open(emb_path, 'wb') as file:
        pkl.dump([z_basal_ds, z_style, y_true_ds_drugs], file)

    return dict_metrics

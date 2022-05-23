"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import sys
sys.path.insert(0, '../..')

import os
import shutil
from collections import OrderedDict
from attr import fields_dict
from tqdm import tqdm
from os.path import join as ospj

import numpy as np
import torch
import pickle as pkl

from compert.core.utils import save_image 
# from core.utils import save_image 

from .fid import calculate_fid_given_images
from .lpips import calculate_lpips_given_images
from .disentanglement_score import compute_disentanglement_score

def calculate_rmse_and_disentanglement_score(nets, 
                                            loader, 
                                            device, 
                                            dest_dir,
                                            embedding_path, 
                                            end,
                                            args, 
                                            step):

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
        y_true_ds_drugs.append(y)  # Record the labels 
        # Draw random vector 
        z = torch.randn(X.shape[0], args.latent_dim)

        with torch.no_grad():
            # We generate in basal mode, so without really conditining on s_trg
            s = None
            z_basal, x_rec_basal = nets.generator(X, s, basal=True)
            # We generate in normal mode
            s = nets.mapping_network(z, y)
            _, x_rec_rand = nets.generator(X, s)
            # Reconstruct from style vector 
            s_post = nets.style_encoder(X, y)
            _, x_rec_post = nets.generator(X, s_post)

            if x_rec_rand_to_plot == None and x_rec_basal_to_plot == None:
                x_rec_rand_to_plot = x_rec_rand
                x_rec_basal_to_plot = x_rec_basal
                x_rec_post_to_plot = x_rec_post
        
        # Update the RMSEs
        rmse_basal_full += torch.sqrt(torch.mean((x_rec_post-x_rec_basal)**2))
        rmse += torch.sqrt(torch.mean((x_rec_post-X)**2))

        # Save the basal state for later visualization 
        z_basal_ds.append(z_basal)
        z_style.append(s_post)

    # Transform basal state and labels to tensors
    z_basal_ds = torch.cat(z_basal_ds, dim=0)
    z_style = torch.cat(z_style, dim=0)
    y_true_ds_drugs = torch.cat(y_true_ds_drugs, dim=0).to('cpu').numpy()
    disentanglement_score = compute_disentanglement_score(z_basal_ds, y_true_ds_drugs)

    # Print metrics 
    dict_metrics = {'Disentanglement_score': disentanglement_score, 
                    'rmse': rmse.item()/len(loader),
                    'rmse_basal_full': rmse_basal_full.item()/len(loader)}
    
    # Plot the last reconstructed basal 
    filename_basal = ospj(dest_dir, args.basal_vs_real_folder, '%06d_basal_decoded_latent.png' % (step))
    filename_rec_rand = ospj(dest_dir, args.basal_vs_real_folder, '%06d_random_decoded_latent.png' % (step))
    filename_rec_post = ospj(dest_dir, args.basal_vs_real_folder, '%06d_post_decoded_latent.png' % (step))

    save_image(x_rec_basal_to_plot[:8], 4, filename_basal)
    save_image(x_rec_post_to_plot[:8], 4, filename_rec_rand)
    save_image(x_rec_rand_to_plot[:8], 4, filename_rec_post)

    if end:
        emb_path = ospj(dest_dir, embedding_path, 'embeddings.pkl')
        print(f"Save embeddings at {emb_path}")
        z_basal_ds = z_basal_ds.detach().cpu().numpy()
        z_style = z_style.detach().cpu().numpy()
        with open(emb_path, 'wb') as file:
            pkl.dump([z_basal_ds, z_style, y_true_ds_drugs], file)

    return dict_metrics


@torch.no_grad()
def calaculate_fid_and_lpips(loader, nets, args, step, id2drug):
    print('Calculating evaluation metrics...')
    # Device of data and the model 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # List the available domains in the transformation pool 
    domains = list(id2drug.keys())
    domains.sort()
    num_domains = len(domains)
    print('Number of domains: %d' % num_domains)

    # Collect the results into the lpsips_dict
    lpips_dict = OrderedDict()
    fid_dict = OrderedDict()

    for trg_idx, trg_domain in enumerate(domains):
        # All domains but the target one 
        src_domains = [x for x in domains if x != trg_domain]

        for src_idx, src_domain in enumerate(src_domains):
            # Task is a conversion from a target to a destination domain
            task = '%s2%s' % (id2drug[src_domain], id2drug[trg_domain])

            lpips_values = []  # lpips values for task
            fid_values = []  # fid values for task

            print('Generating images and calculating LPIPS for and FID for task %s...' % task)
            for i, batch in enumerate(tqdm(loader, total=len(loader))):
                # Fetch batch
                x, y = batch['X'], batch['mol_one_hot'].argmax(1)
                # Keep only indices of source domain 
                idx_to_keep_src = [i for i in range(len(y)) if y[i].item()==src_domain]
                x_src = x[idx_to_keep_src,:,:,:].to(device)

                idx_to_keep_target = [i for i in range(len(y)) if y[i].item()==trg_domain]
                x_trg = x[idx_to_keep_target,:,:,:].to(device)
                
                # Batch size 
                N_source, N_target = x_src.size(0), x_trg.size(0)  

                # Generate 10 outputs from the same input
                group_of_images = []
                
                # In case there are no drugs of a type in the batch 
                if N_source == 0 or N_target==0:
                    continue

                for j in range(args.num_outs_per_domain):
                    # Generate fake vector and the associated style 
                    z_trg = torch.randn(N_source, args.latent_dim).to(device)
                    s_trg = nets.mapping_network(z_trg, trg_domain*torch.ones(N_source).to(z_trg.device).long())

                    # Generate a group of fake images 
                    _, x_fake = nets.generator(x_src, s_trg)
                    group_of_images.append(x_fake)

                # Given the images, compute the lpips value 
                lpips_value = calculate_lpips_given_images(group_of_images)
                lpips_values.append(lpips_value)

                # FID value 
                fid_value = calculate_fid_given_images(x_trg, 
                                                        torch.cat(group_of_images, dim=0), 
                                                        task)
                fid_values.append(fid_value)

            # Calculate LPIPS for the task and add to dict 
            lpips_mean = np.array(lpips_values).mean()
            lpips_dict['LPIPS_%s' %  (task)] = lpips_mean

            fid_mean = np.array(fid_values).mean()
            fid_dict['FID_%s' %  (task)] = fid_mean

    # calculate the average LPIPS for all tasks (hence, pairwise translations)
    lpips_mean = 0
    for _, value in lpips_dict.items():
        lpips_mean += value / len(lpips_dict)
    lpips_dict['LPIPS_total_mean'] = lpips_mean

    fid_mean = 0
    for _, value in fid_dict.items():
        fid_mean += value / len(fid_dict)
    fid_dict['FID_total_mean'] = fid_mean 

    # calculate and report fid values
    return lpips_dict, fid_dict 

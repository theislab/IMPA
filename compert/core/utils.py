"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import json
import glob
from shutil import copyfile

from tqdm import tqdm
import ffmpeg

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils

import matplotlib.pyplot as plt


########################### NN UTILS ###########################


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)



########################### PLOTTING FUNCTIONS ###########################


def plot_transformation_channel(x, id2drug, y, y_transf, filename):
    # The plotting will be batch size x number of random vectors
    fig = plt.figure(figsize=(10, 10))
    columns = len(x)
    rows = int(x[0].shape[0])
    y = y.detach().cpu().numpy()  # Original label
    y_transf = y_transf.detach().cpu().numpy()  # Swapped - Used to give names to plots 

    # ax enables access to manipulate each of subplots
    ax = []
    for i in range(rows*columns):
        ax.append(fig.add_subplot(rows, columns, i+1)) 
        # The add_subplot command acts by row, so the row to plot is i//rows and column is i%columns
        row_to_show = i//(rows-1)
        col_to_show = i%columns
        numpy_img = denormalize(x[col_to_show][row_to_show]).detach().cpu().permute((1,2,0)).numpy()
        ax[-1].imshow(numpy_img)
        if col_to_show == 0:
            ax[-1].set_title(id2drug[y[row_to_show]])
        else:
            ax[-1].set_title(id2drug[y_transf[row_to_show]])
        ax[-1].axis("off")
    filename = filename + '_transformations.jpg'
    plt.savefig(filename)


@torch.no_grad()
def debug_image(nets, embedding_matrix, args, inputs, step, device, id2drug, dest_dir):
    # Fetch the real and fake inputs and outputs 
    x_real, y_one_hot = inputs['X'].to(device), inputs['mol_one_hot'].to(device)
    y_real = y_one_hot.argmax(1).to(device)

    # Setup the device and the batch size 
    device = x_real.device
    N = x_real.size(0)

    # Extract id of dmso and put it at the end of the queue
    drug2id = {val:key for key,val in id2drug.items()}
    dmso_id = drug2id['DMSO']

    # Get all the possible output targets
    range_classes = list(range(args.num_domains))
    range_classes = range_classes[:dmso_id] + range_classes[dmso_id+1:] + [dmso_id]  # Put dmso id at the end of the plot 
    y_trg_list = [torch.tensor(y).repeat(N).to(device)
                  for y in range_classes] 

    # Produce broad transformation panel 
    num_transf_plot = 3 if args.lambda_noise>0 else 1  # Plot only one transformation for non-stochastic run 
    z_trg_list = torch.randn(num_transf_plot, 1, args.z_dimension).repeat(1, N, 1).to(device)
    # Encode noise if required
    if args.learn_noise:
        z_trg_list = nets.noise_projector(z_trg_list)    

    # Swap attributes for smaller cross-transformation panel 
    y_swapped_one_hot = swap_attributes(y_drug=y_one_hot, drug_id=y_real, device=device)
    y_swapped = y_swapped_one_hot.argmax(1)  # y_swapped

    for psi in [0.5, 0.7, 1.0]:
        filename = ospj(dest_dir, args.sample_dir, '%06d_latent_psi_%.1f' % (step, psi))
        translate_using_latent(nets,
                                embedding_matrix, 
                                args, 
                                x_real,
                                y_real,
                                y_swapped, 
                                y_trg_list, 
                                z_trg_list, 
                                psi, 
                                filename, 
                                id2drug)


@torch.no_grad()
def translate_using_latent(nets, 
                            embedding_matrix,
                            args, 
                            x_real,
                            y_real,
                            y_swapped,
                            y_trg_list, 
                            z_trg_list, 
                            psi, 
                            filename, 
                            id2drug):

    # Batch dimension, channel dimension, height and width dimensions 
    N, _, _, _ = x_real.size()
    # Place the validation input in list for concatenation 
    x_concat = [x_real]
    
    # For each domain, collect a latent mean vector 
    for _, y_trg in enumerate(y_trg_list):
        z_many = torch.randn(10000, args.z_dimension).to(x_real.device)
        # Encode noise if required
        if args.learn_noise:
            z_many = nets.noise_projector(z_many)  
        
        y_many = torch.LongTensor(10000).to(x_real.device).fill_(y_trg[0])

        # Add noise to rdkit embedding 
        z_emb_many = embedding_matrix(y_many) 

        s_many = nets.mapping_network(z_emb_many) if args.encode_rdkit else z_emb_many
        s_many += args.lambda_noise*z_many
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(N, 1)  # Have a batch of average response

        for z_trg in z_trg_list:
            z_emb_trg = embedding_matrix(y_trg) 
            s_trg = nets.mapping_network(z_emb_trg) if args.encode_rdkit else z_emb_trg
            s_trg += args.lambda_noise*z_trg
            s_trg = torch.lerp(s_avg, s_trg, psi)
            _, x_fake = nets.generator(x_real, s_trg)
            x_concat += [x_fake]
    
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)  # SAVE LARGE PANEL OF TRANSFORMATIONS 
    
    # For translation plot, keep a batch of size 5
    x_concat = [x_real[:5]]
    
    # Only for psi = 1.0
    if psi == 1.0:
        # Plot the single transformations
        for z_trg in z_trg_list:  # Only a certain number of targets is kept 
            z_emb_trg = embedding_matrix(y_swapped[:5]) 
            s_trg = nets.mapping_network(z_emb_trg) if args.encode_rdkit else z_emb_trg 
            s_trg += args.lambda_noise*z_trg[:5]
            _, x_fake = nets.generator(x_real[:5], s_trg)
            x_concat += [x_fake]

        plot_transformation_channel(x_concat, id2drug, y_real, y_swapped, filename)


########################### EXTRA UTILS ###########################

def print_metrics(metrics_dict, step):
    for metric in metrics_dict:
        print('After %i %s is %.4f' % (step, metric, metrics_dict[metric]))

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename+'.jpg', nrow=ncol, padding=0)

def swap_attributes(y_drug, drug_id, device):
    # Initialize the swapped indices 
    swapped_idx = torch.zeros_like(y_drug)
    # Maximum drug index
    max_drug = y_drug.shape[1] 
    # Ranges of possible drugs 
    offsets = torch.randint(1, max_drug, (drug_id.shape[0], 1)).to(device)
    # Permute
    permutation = drug_id + offsets.squeeze()
    # Remainder 
    permutation = torch.remainder(permutation, max_drug)
    # Add ones 
    swapped_idx[np.arange(y_drug.shape[0]), permutation] = 1
    return swapped_idx

def sigmoid(x, w=1):
    return 1. / (1 + np.exp(-w * x))

def get_alphas(start=-5, end=5, step=0.5, len_tail=10):
    return [0] + [sigmoid(alpha) for alpha in np.arange(start, end, step)] + [1] * len_tail

def interpolate(nets, args, x_src, s_prev, s_next):
    ''' returns T x C x H x W '''
    B = x_src.size(0)
    frames = []
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    alphas = get_alphas()

    for alpha in alphas:
        s_ref = torch.lerp(s_prev, s_next, alpha)
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        entries = torch.cat([x_src.cpu(), x_fake.cpu()], dim=2)
        frame = torchvision.utils.make_grid(entries, nrow=B, padding=0, pad_value=-1).unsqueeze(0)
        frames.append(frame)
    frames = torch.cat(frames)
    return frames

def tensor2ndarray255(images):
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    return images.cpu().numpy().transpose(0, 2, 3, 1) * 255
    
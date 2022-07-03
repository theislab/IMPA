from inspect import ArgInfo
from os.path import join as ospj
import json
from shutil import copyfile

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import itertools


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

def flatten_channels(batch):
    x_concat = list(batch.split(split_size=1, dim=0))
    x_concat =[list(x.split(split_size=1, dim=1)) for x in x_concat]
    x_concat = list(itertools.chain(*x_concat))
    return x_concat


########################### PLOTTING FUNCTIONS ###########################

@torch.no_grad()
def debug_image(nets, embedding_matrix, args, inputs, step, device, id2mol, dest_dir):
    # Fetch the real and fake inputs and outputs 
    x_real, y_one_hot = inputs['X'].to(device), inputs['mol_one_hot'].to(device)
    y_real = y_one_hot.argmax(1).to(device)

    # Setup the device and the batch size 
    device = x_real.device
    N = x_real.size(0)

    # Get all the possible output targets
    range_classes = list(range(args.num_domains))
    y_trg_list = [torch.tensor(y).repeat(N).to(device)
                  for y in range_classes] 
    # Only keep 6 classes 
    y_trg_list = y_trg_list[:6]  
    if len(y_trg_list)>6:
        y_trg_list.shuffle()

    # Produce broad transformation panel 
    num_transf_plot = 1 if not args.stochastic else 2 # Plot only one transformation for non-stochastic run 

    # Create the noise vector 
    z_trg_list = torch.randn(num_transf_plot, 1, args.z_dimension).repeat(1, N, 1).to(device)  

    # Swap attributes for smaller cross-transformation panel 
    y_swapped_one_hot = swap_attributes(y_mol=y_one_hot, mol_id=y_real, device=device)
    y_swapped = y_swapped_one_hot.argmax(1)  # y_swapped

    filename = ospj(dest_dir, args.sample_dir, '%06d_latent' % (step))
    translate_using_latent(nets,
                            embedding_matrix, 
                            args, 
                            x_real,
                            y_real,
                            y_swapped, 
                            y_trg_list, 
                            z_trg_list,
                            filename, 
                            id2mol)


@torch.no_grad()
def translate_using_latent(nets, 
                            embedding_matrix,
                            args, 
                            x_real,
                            y_real,
                            y_swapped,
                            y_trg_list, 
                            z_trg_list,
                            filename, 
                            id2mol):

    # Batch dimension, channel dimension, height and width dimensions 
    N, _, _, _ = x_real.size()
    # Place the validation input in list for concatenation 
    if args.dataset_name == 'bbbc021':
        x_concat = [x_real]
        ncol = N
    else:
        x_concat = flatten_channels(x_real)
        ncol = N*x_real.shape[1]

    # For each domain, collect a latent mean vector 
    for _, y_trg in enumerate(y_trg_list):
        for z_trg in z_trg_list:

            # RDKit embedding
            z_emb_trg = embedding_matrix(y_trg) 

            if args.stochastic:
                z_emb_trg = torch.cat([z_emb_trg, z_trg], dim=1)

            # Perform mapping 
            s_trg = nets.mapping_network(z_emb_trg) if args.encode_rdkit else z_emb_trg

            _, x_fake = nets.generator(x_real, s_trg)
            if args.dataset_name == 'bbbc021':
                x_concat += [x_fake]
            else:
                x_concat += flatten_channels(x_fake)
    
    x_concat = torch.cat(x_concat, dim=0)
    # Large panel of transformations 
    save_image(x_concat, ncol, filename) 


########################### EXTRA UTILS ###########################

def print_metrics(metrics_dict, step):
    for metric in metrics_dict:
        print('After %i %s is %.4f' % (step, metric, metrics_dict[metric]))

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def save_image(x, ncol, filename):
    print(x.shape)
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename+'.jpg', nrow=ncol, padding=0)

def swap_attributes(y_mol, mol_id, device):
    # Initialize the swapped indices 
    swapped_idx = torch.zeros_like(y_mol)
    # Maximum mol index
    max_mol = y_mol.shape[1] 
    # Ranges of possible mols 
    offsets = torch.randint(1, max_mol, (mol_id.shape[0], 1)).to(device)
    # Permute
    permutation = mol_id + offsets.squeeze()
    # Remainder 
    permutation = torch.remainder(permutation, max_mol)
    # Add ones 
    swapped_idx[np.arange(y_mol.shape[0]), permutation] = 1
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

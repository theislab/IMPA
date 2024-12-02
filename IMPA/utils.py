import datetime
import time
from os.path import join as ospj

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

def print_network(network, name):
    """Print the number of parameters in a neural network.

    Args:
        network (torch.nn.Module): The neural network whose parameters are printed.
        name (str): Name of the model.
    """
    num_params = sum(p.numel() for p in network.parameters())
    print(f"Number of parameters in {name}: {num_params}")

def he_init(module):
    """Initialize a network module using the He initialization method.

    Args:
        module (torch.nn.Module): The network module to initialize.
    """
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def print_checkpoint(i, start_time, total_iters, all_losses):
    """Print the elapsed time and losses during training.

    Args:
        i (int): Iteration number.
        start_time (int): Start time in seconds.
        total_iters (int): Total number of iterations.
        all_losses (dict): Dictionary with the loss values.
    """
    elapsed = time.time() - start_time
    elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
    log = f"Elapsed time [{elapsed}], Iteration [{i+1}/{total_iters}], "
    log += ' '.join([f'{key}: [{value:.4f}]' for key, value in all_losses.items()])
    print(log)

def print_metrics(metrics_dict, step):
    """Print metrics from a metric dictionary.

    Args:
        metrics_dict (dict): Dictionary with the metrics.
        step (int): Current step.
    """
    for metric, value in metrics_dict.items():
        print(f'After {step} steps, {metric} is {value:.4f}')

def denormalize(x):
    """Denormalize an image from the range (-1, 1) to (0, 1).

    Args:
        x (torch.Tensor): Input image in the range (-1, 1).

    Returns:
        torch.Tensor: Denormalized image in the range (0, 1).
    """
    out = (x + 1.) / 2
    return out.clamp_(0, 1)

def save_image(x, ncol, filename):
    """Save a panel of images.

    Args:
        x (torch.Tensor): Tensor representing a grid of images.
        ncol (int): Number of columns for plotting.
        filename (str): Name of the file to save.
    """
    print(x.shape)
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename + '.jpg', nrow=ncol, padding=0)

def swap_attributes(max_mol, mol_id, device):
    """Perform random swapping of perturbation categories in a target dataset.

    Args:
        y_mol (torch.Tensor): One-hot array representing the identity of each observation.
        mol_id (torch.Tensor): An array of indexes representing the perturbation used for each observation.
        device (str): 'cuda' or 'cpu'.

    Returns:
        torch.Tensor: The swapped tensor.
    """
    offsets = torch.randint(1, max_mol, (mol_id.shape[0], 1)).to(device)
    permutation = (mol_id + offsets.squeeze()) % max_mol
    return permutation

def sigmoid(x, w=1):
    """Sigmoid function.

    Args:
        x (torch.Tensor): Input of the sigmoid function.
        w (int, optional): Weight to multiply the input. Defaults to 1.

    Returns:
        torch.Tensor: Result of the sigmoid function application.
    """
    return 1. / (1 + np.exp(-w * x))

def tensor2ndarray255(images):
    """Convert tensor to an 8-bit numpy array.

    Args:
        images (torch.Tensor): Images to convert to an array.

    Returns:
        numpy.array: Images converted to a numpy array.
    """
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    return images.cpu().numpy().transpose(0, 2, 3, 1) * 255

@torch.no_grad()
def debug_image(solver, nets, embedding_matrix, inputs, step, device, dest_dir, num_domains, multimodal=False, mod_list=None):
    """Dump a grid of generated images.

    Args:
        solver (object): The solver object containing the training logic.
        nets (dict): Dictionary with the neural network modules.
        embedding_matrix (torch.Tensor): Perturbation embeddings.
        inputs (torch.Tensor): Input used to produce the grid image.
        step (int): The step at which the images were produced.
        device (str): 'cuda' or 'cpu'.
        dest_dir (str): Destination directory for images.
        num_domains (int): Number of target domains.
        batch_correction (bool): Whether batch correction is applied.
        multimodal (bool, optional): Whether to use multimodal data. Defaults to False.
        mod_list (list, optional): List of modalities. Defaults to None.
    """
    # Get the images and the perturbation targets
    if solver.args.batch_correction:
        x_real = inputs['X'].to(device)
    else:
        x_real, _ = inputs['X']
        x_real = x_real.to(device)
        
    # Number of observations 
    N = x_real.size(0)
    
    if solver.args.batch_correction or (not solver.args.batch_correction and not solver.args.multimodal):
        # Get all the possible output targets
        range_classes = list(range(num_domains))
        y_trg_list = [torch.tensor(y).repeat(N).to(device) for y in range_classes] 
        
        # Only keep a maximum of 6 classes at random
        y_trg_list = y_trg_list[:5]  

        # Produce two plots per perturbation
        num_transf_plot = 2 

        # Create the noise vector 
        z_trg_list = torch.randn(num_transf_plot, 1, solver.args.z_dimension).repeat(1, N, 1).to(device)  # 2 x N x D 
        
        y_mod = None   
    
    else:
        y_trg_list = []
        y_mod = []
        for i, _ in enumerate(mod_list):
            y_trg = [torch.tensor(y).repeat(N).to(device) for y in range(4)]
            y_trg_list += y_trg
            y_mod += [i*torch.ones(N).long().cuda()]*4
        
        z_trg_list = None    

    filename = ospj(dest_dir, solver.args.sample_dir, '%06d_latent' % (step))
    translate_using_latent(solver, nets, embedding_matrix, solver.args, x_real, y_trg_list, z_trg_list, filename, y_mod)

@torch.no_grad()
def translate_using_latent(solver, nets, embedding_matrix, args, x_real, y_trg_list, z_trg_list, filename, y_mod=None):
    """Generate and save translated images using latent vectors.

    Args:
        solver (object): The solver object containing the training logic.
        nets (dict): Dictionary with the neural network modules.
        embedding_matrix (torch.Tensor): Perturbation embeddings.
        args (dict): Training specifications.
        x_real (torch.Tensor): Tensor of real images.
        y_trg_list (list): List of target perturbations.
        z_trg_list (list): List of noise vectors for the transformation.
        filename (str): Name of the file to save.
        batch_correction (bool, optional): Whether batch correction is applied. Defaults to False.
        y_mod (list, optional): List of modalities. Defaults to None.
    """
    N, _, _, _ = x_real.size()
    x_concat = [x_real]
    
    for _, y_trg in enumerate(y_trg_list):
        if solver.args.batch_correction or (not solver.args.batch_correction and not solver.args.multimodal):
            for z_trg in z_trg_list:
                z_emb_trg = embedding_matrix(y_trg) 
                z_emb_trg = torch.cat([z_emb_trg, z_trg], dim=1)                
                
                s_trg = nets.mapping_network(z_emb_trg) 
                _, x_fake = nets.generator(x_real, s_trg)
                x_concat += [x_fake]
        else:
            for i, y_trg in enumerate(y_trg_list):        
                if solver.args.multimodal:
                    x_real, s_trg, _, _ = solver.encode_label(x_real, y_trg, y_mod[i], 3)
                else:
                    s_trg = solver.encode_label(x_real, y_trg, None, None)
                _, x_fake = nets.generator(x_real, s_trg)
                x_concat += [x_fake]
        
    x_concat = torch.cat(x_concat, dim=0)
    if args.dataset_name == 'bbbc021':
        save_image(x_concat, N, filename) 
    elif args.dataset_name == 'bbbc025' or ('cpg0000'in args.dataset_name):
        save_image(x_concat[:, [1, 3, 4], :, :], N, filename)
    else:
        save_image(x_concat[:, [5, 1, 0], :, :], N, filename)
    
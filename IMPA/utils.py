import datetime
import time
from os.path import join as ospj

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils


def print_network(network, name):
    """Prints neural network parameters.

    Args:
        network (torch.nn.Module): The neural network whose parameters are printed.
        name (str): Name of the model.
    """
    num_params = sum(p.numel() for p in network.parameters())
    print("Number of parameters in %s: %i" % (name, num_params))


def he_init(module):
    """Initialize neural network using the He initialization method.

    Args:
        module (torch.nn.Module): The network module to initialize.
    """
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def print_checkpoint(i, start_time, total_iters, all_losses):
    """Print elapsed time and losses during training.

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
        print(f'After {step} {metric} is {value:.4f}')


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


def swap_attributes(y_mol, mol_id, device):
    """Perform random swapping of perturbation categories in a target dataset.

    Args:
        y_mol (torch.Tensor): One-hot array representing the identity of each observation.
        mol_id (torch.Tensor): An array of indexes representing the perturbation used for each observation.
        device (str): 'cuda' or 'cpu'.

    Returns:
        torch.Tensor: The swapped tensor.
    """
    # Initialize the swapped indices
    swapped_idx = torch.zeros_like(y_mol)
    # Maximum mol index
    max_mol = y_mol.shape[1]
    # Ranges of possible mols
    offsets = torch.randint(1, max_mol, (mol_id.shape[0], 1)).to(device)
    # Permute
    permutation = (mol_id + offsets.squeeze()) % max_mol
    # Add ones
    swapped_idx[np.arange(y_mol.shape[0]), permutation] = 1
    return swapped_idx


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
def debug_image(nets, embedding_matrix, args, inputs, step, device, id2mol, dest_dir):
    """Dump a grid of generated images.

    Args:
        nets (dict): Dictionary with the neural network modules.
        embedding_matrix (torch.Tensor): Perturbation embeddings.
        args (dict): Training specifications.
        inputs (torch.Tensor): Input used to produce the grid image.
        step (int): The step at which the images were produced.
        device (str): 'cuda' or 'cpu'.
        id2mol (dict): Dictionary mapping identification number to molecule.
        dest_dir (str): Destination directory for images.
    """
    # Get the images and the perturbation targets
    x_real, y_one_hot = inputs['X'].to(device), inputs['mol_one_hot'].to(device)
    y_real = y_one_hot.argmax(1).to(device)

    # Setup the device and the batch size
    N = x_real.size(0)
    # Get all the possible output targets
    range_classes = list(range(args.num_domains))
    y_trg_list = [torch.tensor(y).repeat(N).to(device)
                  for y in range_classes] 
    
    # Only keep 6 classes at random
    y_trg_list = y_trg_list[:6]  
    if len(y_trg_list)>6:
        y_trg_list.shuffle()

    # Produce two plots per perturbation
    num_transf_plot = 2 

    # Create the noise vector 
    if args.stochastic:
        z_trg_list = torch.randn(num_transf_plot, 1, args.z_dimension).repeat(1, N, 1).to(device)  
    else:
        z_trg_list = [None]*num_transf_plot

    # Swap attributes for smaller cross-transformation panel 
    y_swapped_one_hot = swap_attributes(y_mol=y_one_hot, mol_id=y_real, device=device)

    filename = ospj(dest_dir, args.sample_dir, '%06d_latent' % (step))
    translate_using_latent(nets,
                            embedding_matrix, 
                            args, 
                            x_real,
                            y_trg_list, 
                            z_trg_list,
                            filename)

@torch.no_grad()
def translate_using_latent(nets, 
                            embedding_matrix,
                            args, 
                            x_real,
                            y_trg_list, 
                            z_trg_list,
                            filename):
    """
    Collect images for the translation 

    Args:
        nets (dict): the dictionary with the neural network modules
        embedding_matrix (torch.Tensor): perturbation embeddings 
        args (dict): the training specification
        x_real (torch.Tensor): the tensor of real images
        y_real (torch.Tensor): 1D tensor with original perturbation labels
        y_swapped (torch.Tensor): 1D tensor with swapped perturbation labels
        y_trg_list (toch.Tensor): 1D tensor with target perturbations
        z_trg_list (torch.Tensor): tensor with noise vectors for the transformation
        filename (str): name of the file to save
        id2mol (dict): dictionary mapping identification number to molecule 
    """
    # Batch dimension, channel dimension, height and width dimensions 
    N, _, _, _ = x_real.size()
    # Place the validation input in list for concatenation 
    x_concat = [x_real]
    
    for _, y_trg in enumerate(y_trg_list):
        for z_trg in z_trg_list:

            # Perturbation embedding and style embedding 
            z_emb_trg = embedding_matrix(y_trg) 
            if args.stochastic:
                z_emb_trg = torch.cat([z_emb_trg, z_trg], dim=1)                
            
            # Perform mapping 
            s_trg = nets.mapping_network(z_emb_trg) 
            _, x_fake = nets.generator(x_real, s_trg)
            x_concat += [x_fake]
    
    # Save images with specific channels depending on the dataset
    x_concat = torch.cat(x_concat, dim=0)
    if args.dataset_name == 'bbbc021':
        save_image(x_concat, N, filename) 
    elif args.dataset_name == 'bbbc025':
        save_image(x_concat[:,[1,3,4],:,:], N, filename)
    else :
        save_image(x_concat[:,[5,1,0],:,:], N, filename)

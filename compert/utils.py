import numpy as np
import os
import pandas as pd
import cv2
from datetime import datetime 
import torch
import torch.nn.functional as F

    
def img_resize(image, width:int, height:int, interpolation:str):
    """Resize input image 

    Args:
        image (np.array): numpy array of dimension (height, width, channels)
        width (int): target width
        height (int): target height 
        interpolation (str): string in (nn, linear, area, cubic, lanczos)


    Returns:
        np.array: Resized image
    """
    interp_dict = {'nn': cv2.INTER_NEAREST, 'linear': cv2.INTER_LINEAR , 'area': cv2.INTER_AREA , 
                   'cubic': cv2.INTER_CUBIC, 'lanczos': cv2.INTER_LANCZOS4 }
    assert (interpolation in interp_dict), 'Unsupported interpolation method'
    im_resized = cv2.resize(image, (width, height), interpolation = cv2.INTER_CUBIC)
    return im_resized


def img_crop(image, width:int, height:int):
    """Crop image in a squared window around its centre

    Args:
        image (np.array): numpy array of dimension (height, width, channels)
        width (int): target image width
        height (int): target image height 

    Returns:
        np.array: Cropped image
    """
    (cx, cy) = image.shape[0]//2, image.shape[1]//2
    img_cropped = image[(cx-width//2):(cx+width//2), (cy-width//2):(cy+width//2), :]
    return img_cropped


def resize_images(data_dir:str, outdir:str, width=64, height=64, interpolation='cubic'):
    """Resize images in a tar file to a pre-defined size 

    Args:
        data_dir (str): name of the directory storing the data
        outdir (str): name of destination directory
        width (int, optional): target width. Defaults to 64.
        height (int, optional):  target height. Defaults to 64.
        interpolation (str, optional): string in (nn, linear, area, cubic, lanczos). Defaults to 'cubic'.
    """
    assert os.path.exists(data_dir), "The data directory doesn't exist" 
    # Move to working directory
    os.chdir(data_dir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)        
    for filename in os.listdir(data_dir):
        # Load the image file 
        img = np.load(os.path.join(data_dir, filename))['sample']
        resized_img = img_resize(img, width, height, interpolation)
        # Save final object
        np.savez(os.path.join(outdir, filename), resized_img)

        
def extract_filenames_and_molecules(input_path:str, labels_path:str, data_path:str, fold:list, save=True, ds_type = 'integrated'): 
    """
    From a subset of the dataset, extract the filenames and the molecules that associate
    to a data fold (train, test, validation)
    -------------------
    index_path: path to the index csv 
    data_path: path to the data folder
    fold: the fold between train, test and val
    save: True if the filename, the molecule names and the smiles should be saved to datapath
    """
    # Load data matrices
    fold_dataset = pd.read_csv(input_path)  # Load the data matrix
    labels = np.load(labels_path, allow_pickle=True)  # Load the assay labels 

    sample_names = fold_dataset.SAMPLE_KEY.values
    molecule_names = fold_dataset.CPD_NAME.values
    molecule_SMILE = fold_dataset.SMILES.values
    molecule_n_cells = fold_dataset.IMG_CNT_CELLS.values
    
    if ds_type == 'integrated':
        molecule_state = fold_dataset.STATE

    if save:
        if ds_type == 'original':
            np.savez(os.path.join(data_path, f'{fold}_data_index'), filenames = sample_names, mol_names = molecule_names, mol_smiles = molecule_SMILE, 
                        assay_labels=labels['assay_labs'], n_cells=molecule_n_cells)
        else:
            # If the data contains as well the DMSO samples, then we return as well the condition between trt and ctr 
            np.savez(os.path.join(data_path, f'{fold}_data_index'), filenames = sample_names, mol_names = molecule_names, mol_smiles = molecule_SMILE, 
                        assay_labels=labels['assay_labs'], state = molecule_state, n_cells=molecule_n_cells)           
    return  sample_names, molecule_names, molecule_SMILE, labels


def tensor_to_image(tensor, batch_first = True):
    """
    Convert tensor to numpy for plotting

    Args:
        tensor (torch.tensor): tensor to be converted to image
        batch_first (bool, optional): Whether the batch comes in first position in the input. Defaults to True.

    Returns:
        np.array: A numpy array representing an image
    """
    if batch_first:
        tensor = tensor.squeeze(0)
    tensor = tensor.permute(1,2,0).to('cpu').detach()
    return tensor.numpy()


def make_dirs(path, experiment_name):
    """Creates result directories for the models

    Args:
        path (str): path where the directory of the experiment should be dumped 
        experiment_name (str): the name of the experiment performed 

    Returns:
        str: name of the destination directory
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")
    if not os.path.exists(path):
        os.mkdir(path)
    dest_dir = os.path.join(path, experiment_name+'_'+timestamp)
    os.mkdir(dest_dir)
    os.mkdir(os.path.join(dest_dir, 'reconstructions'))
    os.mkdir(os.path.join(dest_dir, 'generations'))
    os.mkdir(os.path.join(dest_dir, 'checkpoints'))
    # os.mkdir(os.path.join(dest_dir, 'logs'))
    return dest_dir


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + torch.nn.functional.softplus(tensor - min)
    return result_tensor


# Auxiliary functions 
def gaussian_nll(mu, log_sigma, x):
    """
    Implement Gaussian nll loss
    """
    return 0.5 * (torch.pow((x - mu) , 2)/ log_sigma.exp() + log_sigma) + 0.5 * np.log(2 * np.pi)


def configure_loss_dict(loss_dict):
    losses = {}
    for key in loss_dict:
        if type(loss_dict[key]) == dict:
            for subkey in loss_dict[key]:
                if loss_dict[key][subkey] != 0:
                    losses[subkey] = loss_dict[key][subkey].item()
        else:
            if loss_dict[key] != 0:
                losses[key] = loss_dict[key].item()
    return losses

"""
Funtion to swap the labels randomly
"""
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

"""
Loss function designed for label smoothing 
"""
# From https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/166833#930136

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1, reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def linear_combination(self, x, y):
        return self.epsilon * x + (1 - self.epsilon) * y

    def forward(self, preds, target):
        # Move weights to device
        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        if self.training:
            n = preds.size(-1)
            log_preds = F.log_softmax(preds, dim=-1)
            loss = self.reduce_loss(-log_preds.sum(dim=-1))
            nll = F.nll_loss(
                log_preds, target, reduction=self.reduction, weight=self.weight
            )
            return self.linear_combination(loss / n, nll)
        else:
            return torch.nn.functional.cross_entropy(preds, target, weight=self.weight)

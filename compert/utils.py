import tarfile
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import timeit,time
import cv2
import torch
from torch import nn
import time 

    
def img_resize(image, width:int, height:int, interpolation:str):
    """
    Resize input image 
    -------------------
    image: numpy array of dimension (height, width, channels)
    width: target width
    height: target height 
    interpolation: string in (nn, linear, area, cubic, lanczos)
    """
    interp_dict = {'nn': cv2.INTER_NEAREST, 'linear': cv2.INTER_LINEAR , 'area': cv2.INTER_AREA , 
                   'cubic': cv2.INTER_CUBIC, 'lanczos': cv2.INTER_LANCZOS4 }
    assert (interpolation in interp_dict), 'Unsupported interpolation method'
    im_resized = cv2.resize(image, (width, height), interpolation = cv2.INTER_CUBIC)
    return im_resized

def img_crop(image, width:int, height:int):
    """
    Crop image in a squared window around its centre
    -------------------
    image: numpy array of dimension (height, width, channels)
    width: target image width
    height: target image height 
    """
    (cx, cy) = image.shape[0]//2, image.shape[1]//2
    img_cropped = image[(cx-width//2):(cx+width//2), (cy-width//2):(cy+width//2), :]
    return img_cropped

def resize_images(data_dir:str, outdir:str, width=64, height=64, interpolation='cubic'):
    """
    Resize images in a tar file to a pre-defined size 
    -------------------
    data_dir: name of the directory storing the data
    outdir: name of destination directory
    width: target width
    height: target height 
    interpolation: string in (nn, linear, area, cubic, lanczos)
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
    
    if ds_type == 'integrated':
        molecule_state = fold_dataset.STATE

    if save:
        if ds_type == 'original':
            np.savez(os.path.join(data_path, f'{fold}_data_index'), filenames = sample_names, mol_names = molecule_names, mol_smiles = molecule_SMILE, 
                        assay_labels=labels['assay_labs'])
        else:
            # If the data contains as well the DMSO samples, then we return as well the condition between trt and ctr 
            np.savez(os.path.join(data_path, f'{fold}_data_index'), filenames = sample_names, mol_names = molecule_names, mol_smiles = molecule_SMILE, 
                        assay_labels=labels['assay_labs'], state = molecule_state)           
    return  sample_names, molecule_names, molecule_SMILE, labels

    
def tensor_to_image(tensor, batch_first = True):
    """
    Convert tensor to numpy for plotting
    """
    if batch_first:
        tensor = tensor.squeeze(0)
    tensor = tensor.permute(1,2,0).to('cpu').detach()
    return tensor.numpy()


def make_dirs(path, experiment_name):
    """
    Creates result directories for the models
    --------------------
    path: path where the directory of the experiment should be dumped 
    experiment_name: the name of the experiment performed 
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(path):
        os.mkdir(path)
    dest_dir = os.path.join(path, experiment_name+'_'+timestamp)
    os.mkdir(dest_dir)
    os.mkdir(os.path.join(dest_dir, 'reconstructions'))
    os.mkdir(os.path.join(dest_dir, 'generations'))
    os.mkdir(os.path.join(dest_dir, 'checkpoints'))
    os.mkdir(os.path.join(dest_dir, 'logs'))
    return dest_dir


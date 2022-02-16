import tarfile
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import timeit,time
import cv2

def plot_channel_panel(image):
    """
    Plot a panel of single channel figures 
    -------------------
    image: numpy array of dimension (height, width, 5)
    """
    fig, axs = plt.subplots(1, 5, figsize = (30,30))
    for z in range(image.shape[-1]):
        axs[z].imshow(image[:,:,z], cmap = 'Greys')
    plt.show()
    
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

def extract_filenames_and_molecules(index_path:str, data_path:str, split:[1,2,3], fold :['train', 'val', 'test'], save=True): 
    """
    From a subset of the dataset, extract the filenames and the molecules that associate
    to a data fold (train, test, validation)
    -------------------
    index_path: path to the index csv 
    data_path: path to the data folder
    fold: the fold between train, test and val
    save: True if the filename, the molecule names and the smiles should be saved to datapath
    """
    # Read the fold dataset 
    fold_dataset== pd.read_csv(os.path.join(index_path, f'datasplit{str(split)}-{fold}.csv'))
    # Extract the sample names, molecule names and the molecule smiles 
    sample_names = fold_dataset.SAMPLE_KEY.values
    molecule_names = fold_dataset.CPD_NAME.values
    molecule_SMILE = fold_dataset.SMILES.values
    # Only keep the sample names and molecules that are present in the folder 
    index = np.arange(len(sample_names))
    filenames = os.listdir(data_path)
    idx_to_keep = [i for i in index if sample_names[i]+'.npz' in filenames]
    # Filter the sample names, molecule names and SMILE names 
    filenames, mol_names, SMILES =  sample_names[idx_to_keep], molecule_names[idx_to_keep], molecule_SMILE[idx_to_keep]
    if save:
        np.savez(os.path.join(data_path, f'{fold}_data_index'), filenames = filenames, mol_names = mol_names, mol_smiles = SMILES)
    return  filenames, mol_names, SMILES


def get_files_and_mols_from_path(data_index_path): 
    """
    Load object with image names, molecule names and smiles 
    -------------------
    data_index_path: The path to the data index with information about molecules and sample names
    """
    assert os.path.exists(data_index_path), 'The data index file does not exist'
    # Load the index file 
    file = np.load(data_index_path, allow_pickle= True)
    file_names, mol_names, mol_smiles = file['filenames'], file['mol_names'], file['mol_smiles']
    return file_names, mol_names, mol_smiles
    
def tensor_to_image(tensor, batch_first = True):
    """
    Convert tensor to numpy for plotting
    """
    if batch_first:
        tensor = tensor.squeeze(0)
    tensor = tensor.permute(1,2,0).to('cpu')
    return tensor.numpy()
    

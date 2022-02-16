"""
Implement custom dataset 
"""

from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
from utils import *

class CustomTransform:
    """
    Scale and resize an input image 
    """
    def __init__(self, normalize = True, test = False):
        self.normalize = normalize
        self.test = test 
        
    def compute_mean_std(self, X, n_channels):
        X_res = X.view(n_channels, -1)
        return X_res.mean(1), X_res.std(1)
        
    def __call__(self, X):
        """
        Transform the image at the time of loading by resizing and turning into a tensor 
        """
        # Compute the mean per channel 
        n_channels = X.shape[0]
        # Compute means and std per channels
        if self.normalize:
            means, std = self.compute_mean_std(X, n_channels)
        # Rescale pixels between 0 and 1 
        X = X/255.0
        if not self.test:
        # Apply resize 
            transform = T.Compose([
                T.Normalize(means, std),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(np.random.randint(0,30)),
                T.RandomVerticalFlip(p=0.5)]
            )
            return transform(X)
        else:
            return X 
    
class CellPaintingDataset(Dataset):
    """
    Dataset class for image data 
    """
    def __init__(self, file_names, mol_names, mol_smiles, data_path, transform):
        """
        data_path: the repository where the data is stored 
        data_index_path: path to .npz object with sample names, molecule names and molecule smiles 
        """
        assert os.path.exists(data_path), 'The data path does not exist'
        self.data_path = data_path
        self.file_names, self.mol_names, self.mol_smiles = file_names, mol_names, mol_smiles
        self.transform = transform
        
    def __len__(self):
        """
        Total number of samples 
        """
        return len(self.file_names)
    
    def __getitem__(self, idx):
        """
        Generate one example datapoint 
        """
        img_file = self.file_names[idx]
        # Load image 
        with np.load(os.path.join(self.data_path, f'{img_file}.npz'), allow_pickle = True) as f:
            #img = f['sample']
            img = f['arr_0']
        img = torch.from_numpy(img).to(torch.float)
        img = img.permute(2,0,1)  # Place channel dimension in front of the others 
        if self.transform != None:
            img = self.transform(img)
        return img

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset

from pytorch_lightning import LightningDataModule
from IMPA.dataset.data_utils import CustomTransform
from IMPA.utils import *


class CellDataset:
    """Dataset class for image data 
    """
    def __init__(self, args, device):
        
        assert os.path.exists(args.image_path), 'The data path does not exist'
        assert os.path.exists(args.data_index_path), 'The data index path does not exist'

        # Set up the variables 
        self.image_path = args.image_path  # Path to the image folder (.pkl file)
        self.data_index_path = args.data_index_path  # Path to data index (.csv file) 
        self.embedding_path = args.embedding_path
        self.augment_train = args.augment_train  
        self.normalize = args.normalize  # Controls whether the input lies between 0 and 1 or -1 and 1
        self.mol_list = args.mol_list
        self.ood_set = args.ood_set  # List of drugs out of distribution
        self.trainable_emb = args.trainable_emb
        self.dataset_name = args.dataset_name 
        self.latent_dim = args.latent_dim

        # Fix the training specifics 
        self.device = device 

        # Read the datasets
        self.fold_datasets = self._read_folds()
        
        # Count the number of compounds 
        trt_dataset = self.fold_datasets['train'].loc[self.fold_datasets['train']["state"]=="control"]
        self.mol_names = np.unique(trt_dataset["CPD_NAME"])  # Sorted drug names
        self.y_names = np.unique(self.fold_datasets['train']["ANNOT"])  # Sorted MOA names (or other annotation) 

        # Count the number of drugs and MOAs 
        self.n_mol = len(self.mol_names) 
        self.n_y = len(self.y_names)

        # Create the embeddings
        if self.trainable_emb:
            self.embedding_matrix = torch.nn.Embedding(self.n_mol, self.latent_dim).to(self.device).to(torch.float32)  # Embedding 
        else:
            embedding_matrix = pd.read_csv(self.embedding_path, index_col=0)
            embedding_matrix = embedding_matrix.loc[self.mol_names]  # Sort based on the drug names 
            embedding_matrix = torch.tensor(embedding_matrix.values, 
                                                    dtype=torch.float32, device=self.device)
            self.embedding_matrix = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=True).to(self.device)

        # Keep track of the indices and numbers 
        self.mol2id = {mol: id for id, mol in enumerate(self.mol_names)}
        self.y2id = {y: id for id, y in enumerate(self.y_names)}

        # Encoders for moa and drug 
        encoder_mol = OneHotEncoder(sparse=False, categories=[self.mol_names])
        encoder_mol.fit(np.array(self.mol_names).reshape((-1,1)))

        # Initialize the datasets 
        self.fold_datasets = {'train': CellDatasetFold('train', 
                                                        self.image_path,
                                                        self.fold_datasets['train'],
                                                        encoder_mol, 
                                                        self.mol2id, 
                                                        self.y2id, 
                                                        self.augment_train, 
                                                        self.normalize),


                                'test': CellDatasetFold('test', 
                                                        self.image_path,
                                                        self.fold_datasets['test'], 
                                                        encoder_mol, 
                                                        self.mol2id, 
                                                        self.y2id, 
                                                        self.augment_train, 
                                                        self.normalize)}                                                    

    def _read_folds(self):
        """Extract the filenames of images in the train and test sets 
        associated folder
        """
        # Read the index csv file
        dataset = pd.read_csv(self.data_index_path, index_col=0)

        # Subset the perturbations if provided in mol_list
        if self.mol_list != None:
            dataset = dataset.loc[dataset.CPD_NAME.isin(self.mol_list)]
        # Remove the leave-out drugs if provided in ood_set
        if self.ood_set!=None:
            dataset = dataset.loc[~dataset.CPD_NAME.isin(self.ood_set)]
        
        # Collect in a dictionary the folds
        dataset_splits = dict()
        
        for fold_name in ['train', 'test']:
            # Divide the dataset in splits 
            dataset_splits[fold_name] = {}

            # Subset of dataframe corresponding to the split 
            subset = dataset.loc[dataset.SPLIT == fold_name]

            for key in subset.columns:
                dataset_splits[fold_name][key] = np.array(subset[key])
                
        return dataset_splits


class CellDatasetFold(Dataset):
    def __init__(self,
                fold, 
                image_path,
                data, 
                encoder_mol, 
                mol2id, 
                y2id, 
                augment_train=True, 
                normalize=False):

        super(CellDatasetFold, self).__init__() 

        # Train or test sets
        self.image_path = image_path
        self.fold = fold  
        self.data = data
        
        # Extract variables 
        self.filenames = {}
        self.mols = {}
        self.y = {}
        
        for cond in ["control", "trt"]:
            # subset by condition 
            data_cond  = data.loc[data.state==cond]
            self.filenames[cond] = data_cond['SAMPLE_KEY']
            self.mols[cond] = data_cond['CPD_NAME']
            self.y[cond] = data_cond['ANNOT']
        
        del data 

        # Whether to perform training augmentation
        self.augment_train = augment_train
        
        # One-hot encoders 
        self.encoder_mol = encoder_mol
        self.mol2id = mol2id
        self.y2id = y2id
        
        # Transform only the training set and only if required
        if self.augment_train and self.fold == 'train':
            self.transform = CustomTransform(augment=True, normalize=normalize)
        else:
            self.transform = CustomTransform(augment=False, normalize=normalize)
        
        if self.fold ==  'train':
            # Map drugs to moas
            mol_ids, y_ids = [self.mol2id[mol] for mol in self.mols["trt"]] , [self.y2id[y] for y in self.y["trt"]]
            self.couples_mol_y = {mol:y for mol,y in zip(mol_ids, y_ids)}

        # One-hot encode molecules and moas
        self.one_hot_mol = self.encoder_mol.transform(np.array(self.mols["trt"].reshape((-1,1))))    

        # Create sampler weights 
        self._get_sampler_weights()

        
    def __len__(self):
        """
        Total number of samples 
        """
        return len(self.file_names["control"])

    def __getitem__(self, idx):
        """
        Generate one example datapoint 
        """
        # Sample control and treated batches 
        idx_trt = np.random.randint(0, len(self.filenames["trt"]))
        img_file_ctrl = self.file_names[idx]
        img_file_trt = self.file_names[idx_trt]
    
        # Split files 
        file_split_ctrl = img_file_ctrl.split('-')
        file_split_trt = img_file_trt.split('-')
        
        if len(file_split_ctrl) > 1:
            file_split_ctrl = file_split_ctrl[1].split("_")
            file_split_trt = file_split_trt[1].split("_")
            path_ctrl = Path(self.image_path) / "_".join(file_split_ctrl[:2]) / file_split_ctrl[2] 
            path_trt = Path(self.image_path) / "_".join(file_split_trt[:2]) / file_split_trt[2] 
            file_ctrl = '_'.join(file_split_ctrl[3:])+".npy"
            file_trt = '_'.join(file_split_trt[3:])+".npy"
        else:
            file_split_ctrl = file_split_ctrl[0].split("_")
            file_split_trt = file_split_trt[0].split("_")
            path_ctrl = Path(self.image_path) / file_split_ctrl[0] / file_split_ctrl[1] 
            path_trt = Path(self.image_path) / file_split_trt[0] / file_split_trt[1] 
            file_ctrl = '_'.join(file_split_ctrl[2:])+".npy"
            file_trt = '_'.join(file_split_trt[2:])+".npy"
            
        img_ctrl, img_trt = np.load(path_ctrl / file_ctrl), np.load(path_trt / file_trt)
        img_ctrl, img_trt = torch.from_numpy(img_ctrl).to(torch.float), torch.from_numpy(img_trt).to(torch.float)
        img_ctrl, img_trt = img_ctrl.permute(2,0,1), img_trt.permute(2,0,1)  # Place channel dimension in front of the others 
        img_ctrl, img_trt = self.transform(img_ctrl), self.transform(img_trt)
        
        return {'X':(img_ctrl, img_trt), 
                'mol_one_hot': self.one_hot_mol[idx_trt], 
                'y_id': self.y2id[self.y[idx_trt]],
                'file_names': (img_file_ctrl, img_file_ctrl)}
    
    def _get_sampler_weights(self):
        mol_names_idx = [self.mol2id[mol] for mol in self.mols["trt"]]
        # Counts and uniqueness 
        mol_names_idx_unique = np.unique(mol_names_idx, return_counts=True)
        dict_mol_names_idx_unique = {key:1/val for key,val in zip(mol_names_idx_unique[0], mol_names_idx_unique[1])}
        # Derive weights
        self.weights = [dict_mol_names_idx_unique[obs] for obs in mol_names_idx]


class CellDataLoader(LightningDataModule):
    """General data loader class
    """
    def __init__(self, args):
        """General argument dictionary 

        Args:
            args (_type_): _description_
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.init_dataset() 
        
    def create_torch_datasets(self):
        """Create dataset compatible with the pytorch training loop 
        """
        dataset = CellDataset(self.args, device=self.device) 
        
        # Channel dimension
        self.dim = self.args['n_channels']

        # Integrate embeddings as class attribute
        self.embedding_matrix = dataset.embedding_matrix  

        # Number of mols and annotations (the latter can be modes of action/genes...)
        self.n_mol = dataset.n_mol
        self.num_y = dataset.n_y 

        # Collect training and test set 
        training_set, test_set = dataset.fold_datasets.values()  
        
        # Collect ids 
        self.mol2id = dataset.mol2id
        self.y2id = dataset.y2id
        self.id2mol = {val:key for key,val in self.mol2id.items()}
        self.id2y = {val:key for key,val in self.y2id.items()}   

        # Free cell painting dataset memory
        del dataset
        return training_set, test_set
        
        
    def init_dataset(self):
        """Initialize dataset and data loaders
        """
        self.training_set, self.test_set = self.create_torch_datasets()
        
        # Create data loaders 
        if self.args.balanced:
            # Balanced sampler
            sampler = WeightedRandomSampler(torch.tensor(self.training_set.weights), len(self.training_set.weights), replacement=False)
            self.loader_train = torch.utils.data.DataLoader(self.training_set, 
                                                            batch_size=self.args.batch_size, 
                                                            sampler=sampler, 
                                                            num_workers=self.args.num_workers, 
                                                            drop_last=True)   
        else:
            self.loader_train = torch.utils.data.DataLoader(self.training_set, 
                                                            batch_size=self.args.batch_size, 
                                                            shuffle=True, 
                                                            num_workers=self.args.num_workers, 
                                                            drop_last=True)  

        self.loader_test = torch.utils.data.DataLoader(self.test_set, 
                                                       batch_size=self.args.val_batch_size, 
                                                       shuffle=False, 
                                                       num_workers=self.args.num_workers,
                                                       drop_last=False)      
        self.mol2y = self.training_set.couples_mol_y       
    
    
    def train_dataloader(self):
        return self.loader_train
    
    def val_dataloader(self):
        return self.loader_test
    
    def val_dataloader(self):
        return self.loader_test
    
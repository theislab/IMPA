"""
Implement custom dataset 
"""
import os
from random import sample
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import pickle as pkl

from ..utils import *

import torch
from torch.utils.data import Dataset
from .data_utils import CustomTransform


class CellDataset:
    """
    Dataset class for image data 
    """
    def __init__(self, args, device):

        assert os.path.exists(args.image_path), 'The data path does not exist'
        assert os.path.exists(args.data_index_path), 'The data index path does not exist'

        # Set up the variables 
        self.image_path = args.image_path  # Path to the image folder (pickle file)
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
        self.mol_names = np.unique(self.fold_datasets['train']["CPD_NAME"])  # Sorted drug names
        self.y_names = np.unique(self.fold_datasets['train']["ANNOT"])  # Sorted MOA names ù

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
        self.fold_datasets = {'train': CellDatasetFold('train', self.fold_datasets['train'], 
                                                        self.image_path, 
                                                        encoder_mol, 
                                                        self.mol2id, 
                                                        self.y2id, 
                                                        self.augment_train, 
                                                        self.normalize),


                            'test': CellDatasetFold('test', self.fold_datasets['test'], 
                                                    self.image_path, 
                                                    encoder_mol, 
                                                    self.mol2id, 
                                                    self.y2id, 
                                                    self.augment_train, 
                                                    self.normalize)}
                                                    

    def _read_folds(self):
        """
        Extract the filenames of images in the train and test sets 
        associated folder
        """
        # Read the index csv file
        dataset = pd.read_csv(self.data_index_path, index_col=0)

        # Only embeddable results
        if not self.trainable_emb and self.dataset_name=='bbbc025':
            dataset = dataset.loc[dataset.CPD_NAME != '0']

        # Remove the out of distribution drugs from the dataset if necessary
        if self.mol_list != None:
            dataset = dataset.loc[dataset.CPD_NAME.isin(self.mol_list)]
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
                # Add the  important entries to the dataset
                dataset_splits[fold_name][key] = np.array(subset[key])
        return dataset_splits


class CellDatasetFold(Dataset):
    def __init__(self, 
                fold, 
                data, 
                image_path, 
                encoder_mol, 
                mol2id, 
                y2id, 
                augment_train=True, 
                normalize=False):

        super(CellDatasetFold, self).__init__() 

        # Train or test sets
        self.fold = fold  
        self.data = data
        
        # Extract variables 
        self.file_names = data['SAMPLE_KEY']
        self.mols = data['CPD_NAME']
        self.dose = data['DOSE']
        self.y = data['ANNOT']

        del data 

        # Image folder paths 
        self.image_path = image_path
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
            mol_ids, y_ids = [self.mol2id[mol] for mol in self.mols] , [self.y2id[y] for y in self.y]
            self.couples_mol_y = {mol:y for mol,y in zip(mol_ids, y_ids)}

        # One-hot encode molecules and moas
        self.one_hot_mol = self.encoder_mol.transform(np.array(self.mols.reshape((-1,1))))    

        # Create sampler weights 
        self._get_sampler_weights()

        # Read images from memory
        print('Loading images...')
        self.read_data()    
        
    def __len__(self):
        """
        Total number of samples 
        """
        return len(self.file_names)
    
    def read_data(self):
        """
        Load images into memory
        """
        with open(self.image_path, 'rb') as file:
            self.image_dict = pkl.load(file)

    def __getitem__(self, idx):
        """
        Generate one example datapoint 
        """
        # Image must be fetched from disk 
        img_file = self.file_names[idx]
        img = self.image_dict[img_file]
        img = torch.from_numpy(img).to(torch.float)
        img = img.permute(2,0,1)  # Place channel dimension in front of the others 
        img = self.transform(img)

        return {'X':img, 
                'mol_one_hot': self.one_hot_mol[idx], 
                'y_id': self.y2id[self.y[idx]],
                'dose': self.dose[idx],
                'file_names': img_file}
    
    def _get_sampler_weights(self):
        mol_names_idx = [self.mol2id[mol] for mol in self.mols]
        # Counts and uniqueness 
        mol_names_idx_unique = np.unique(mol_names_idx, return_counts=True)
        dict_mol_names_idx_unique = {key:1/val for key,val in zip(mol_names_idx_unique[0], mol_names_idx_unique[1])}
        # Derive weights
        self.weights = [dict_mol_names_idx_unique[obs] for obs in mol_names_idx]

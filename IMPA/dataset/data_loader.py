import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from IMPA.dataset.data_utils import CustomTransform, read_files_batch, read_files_pert
from IMPA.utils import *

class CellDataset:
    """
    Dataset class for cell image data.

    This class handles the loading and preprocessing of cell image datasets, 
    including the initialization of dataset splits, normalization, and embedding creation.
    """
    
    def __init__(self, args, device):
        """
        Initialize the CellDataset instance.
        
        Args:
            args (argparse.Namespace): Arguments containing dataset configuration.
            device (torch.device): Device to load the data onto (e.g., 'cuda' or 'cpu').
        """
        assert os.path.exists(args.image_path), 'The data path does not exist'
        assert os.path.exists(args.data_index_path), 'The data index path does not exist'

        # Set up the variables
        self.image_path = args.image_path  # Path to the image folder (.pkl file)
        self.data_index_path = args.data_index_path  # Path to data index (.csv file)
        self.embedding_path = args.embedding_path  # Path to embeddings
        self.augment_train = args.augment_train  # Whether to apply data augmentation during training
        self.normalize = args.normalize  # Controls whether to normalize input images
        self.mol_list = args.mol_list  # List of molecules to include
        self.ood_set = args.ood_set  # List of out-of-distribution drugs
        self.trainable_emb = args.trainable_emb  # Whether embeddings are trainable
        self.dataset_name = args.dataset_name  # Name of the dataset
        self.latent_dim = args.latent_dim  # Dimension of the latent space

        self.batch_correction = args.batch_correction  # If True, perform batch correction
        self.multimodal = args.multimodal  # If True, handle multiple types of perturbations

        if not self.batch_correction:
            self.add_controls = args.add_controls  # Whether to add controls in non-batch correction mode
            self.batch_key = None
        else:
            self.add_controls = None
            self.batch_key = args.batch_key  # Key for batch correction

        # Fix the training specifics
        self.device = device 

        # Read the datasets
        self.fold_datasets = self._read_folds()

        self.y_names = np.unique(self.fold_datasets['train']["ANNOT"])  # Sorted annotation names

        # Count the number of compounds 
        self._initialize_mol_names()

        self.y2id = {y: id for id, y in enumerate(self.y_names)}  # Map annotations to IDs
        self.n_y = len(self.y_names)  # Number of unique annotations
        
        # Initialize embeddings 
        self.initialize_embeddings()

        # Initialize the datasets
        self.fold_datasets = {
            'train': CellDatasetFold('train', 
                                     self.image_path, 
                                     self.fold_datasets['train'],
                                     self.mol2id,
                                     self.y2id, 
                                     self.augment_train, 
                                     self.normalize,
                                     dataset_name=self.dataset_name,
                                     add_controls=self.add_controls, 
                                     batch_correction=self.batch_correction,
                                     batch_key=self.batch_key,
                                     multimodal=self.multimodal,
                                     cpd_name=self.cpd_name),
            
            'test': CellDatasetFold('test',
                                    self.image_path,
                                    self.fold_datasets['test'],
                                    self.mol2id, 
                                    self.y2id, 
                                    self.augment_train, 
                                    self.normalize,
                                    dataset_name=self.dataset_name,
                                    add_controls=self.add_controls,
                                    batch_correction=self.batch_correction,
                                    batch_key=self.batch_key,
                                    multimodal=self.multimodal,
                                    cpd_name=self.cpd_name)}

    def _read_folds(self):
        """
        Extract the filenames of images in the train and test sets.
        
        Returns:
            dict: Dictionary containing train and test datasets.
        """
        # Read the index CSV file
        dataset = pd.read_csv(self.data_index_path, index_col=0)
        
        # Initialize CPD_NAME differently based on the dataset 
        self.cpd_name = "BROAD_SAMPLE" if self.dataset_name == "cpg0000" else "CPD_NAME"

        # Subset the perturbations if provided in mol_list
        if self.mol_list:
            dataset = dataset.loc[dataset[self.cpd_name].isin(self.mol_list)]
        # Remove the leave-out drugs if provided in ood_set
        if self.ood_set is not None:
            dataset = dataset.loc[~dataset[self.cpd_name].isin(self.ood_set)]
        
        # Collect the dataset splits
        dataset_splits = dict()
        
        for fold_name in ['train', 'test']:
            # Divide the dataset in splits 
            dataset_splits[fold_name] = {}
            
            # Divide the dataset into splits
            subset = dataset.loc[dataset.SPLIT == fold_name]
            for key in subset.columns:
                dataset_splits[fold_name][key] = np.array(subset[key])
            if not self.batch_correction:
                # Add control and treated flags
                if not self.add_controls:
                    dataset_splits[fold_name]["trt_idx"] = (dataset_splits[fold_name]["STATE"] == "trt")
                else:
                    dataset_splits[fold_name]["trt_idx"] = (np.isin(dataset_splits[fold_name]["STATE"], ["trt", "control"]))
                dataset_splits[fold_name]["ctrl_idx"] = (dataset_splits[fold_name]["STATE"] == "control")
                
        return dataset_splits

    def _initialize_mol_names(self):
        """
        Initialize molecule names and counts based on dataset splits.
        """
        if not self.batch_correction:
            if not self.multimodal:
                if self.add_controls:
                    self.mol_names = np.unique(self.fold_datasets["train"][self.cpd_name])
                else:
                    self.mol_names = np.unique(self.fold_datasets["train"][self.cpd_name][self.fold_datasets["train"]["trt_idx"]])
                self.n_mol = len(self.mol_names)
            else:
                self.mol_names = {}
                for pert_type in self.y_names:
                    idx_pert = self.fold_datasets["train"]["ANNOT"] == pert_type
                    if self.add_controls:
                        self.mol_names[pert_type] = np.unique(self.fold_datasets["train"][self.cpd_name][idx_pert])
                    else:
                        trt_idx = self.fold_datasets["train"]["trt_idx"][idx_pert]
                        self.mol_names[pert_type] = np.unique(self.fold_datasets["train"][self.cpd_name][idx_pert][trt_idx])
                self.n_mol = {key: len(val) for key, val in self.mol_names.items()} 
        else: 
            self.mol_names = np.unique(self.fold_datasets['train'][self.batch_key])
            self.n_mol = len(self.mol_names)

    def initialize_embeddings(self):
        """
        Create and initialize the embeddings for molecules.
        """
        if self.multimodal and (not self.trainable_emb and not self.batch_correction):
            embedding_matrix = []
            mol2id = {}
            self.latent_dim = {}

            for mod in self.y_names:
                embedding_matrix_modality = pd.read_csv(self.embedding_path[mod], index_col=0)
                embedding_matrix_modality = embedding_matrix_modality.loc[self.mol_names[mod]]
                embedding_matrix_modality = torch.tensor(embedding_matrix_modality.values, dtype=torch.float32, device=self.device)
                self.latent_dim[mod] = embedding_matrix_modality.shape[1]
                embedding_matrix_modality = torch.nn.Embedding.from_pretrained(embedding_matrix_modality, freeze=True).to(self.device)
                embedding_matrix.append(embedding_matrix_modality)
                mol2id[mod] = {mol: id for id, mol in enumerate(self.mol_names[mod])}
                
            self.embedding_matrix = torch.nn.ModuleList(embedding_matrix)
            self.mol2id = mol2id
            
        else:
            if self.trainable_emb or self.batch_correction:
                self.embedding_matrix = torch.nn.Embedding(self.n_mol, self.latent_dim).to(self.device).to(torch.float32)
            else:
                embedding_matrix = pd.read_csv(self.embedding_path, index_col=0)
                embedding_matrix = embedding_matrix.loc[self.mol_names]
                embedding_matrix = torch.tensor(embedding_matrix.values, dtype=torch.float32, device=self.device)
                self.latent_dim = embedding_matrix.shape[1]
                self.embedding_matrix = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=True).to(self.device)
            
            self.mol2id = {mol: id for id, mol in enumerate(self.mol_names)}

class CellDatasetFold(Dataset):
    """
    Dataset fold class for handling train and test splits of cell image data.

    This class inherits from PyTorch's Dataset and provides methods to 
    handle data loading, transformations, and batch processing.
    """
    
    def __init__(self,
                 fold, 
                 image_path, 
                 data, 
                 mol2id,
                 y2id,
                 augment_train=True, 
                 normalize=False, 
                 dataset_name="bbbc021", 
                 add_controls=None,
                 batch_correction=False, 
                 batch_key="BATCH", 
                 multimodal=False, 
                 cpd_name="CPD_NAME"):
        """
        Initialize the CellDatasetFold instance.
        
        Args:
            fold (str): 'train' or 'test' to specify the dataset split.
            image_path (str): Path to the image folder.
            data (dict): Data dictionary containing sample information.
            mol2id (dict): Mapping from molecule names to IDs.
            y2id (dict): Mapping from annotation names to IDs.
            augment_train (bool, optional): Whether to apply data augmentation. Defaults to True.
            normalize (bool, optional): Whether to normalize the images. Defaults to False.
            dataset_name (str, optional): Name of the dataset. Defaults to "bbbc021".
            add_controls (bool, optional): Whether to add controls. Defaults to None.
            batch_correction (bool, optional): Whether to perform batch correction. Defaults to False.
            batch_key (str, optional): Key for batch correction. Defaults to "BATCH".
            multimodal (bool, optional): Whether to handle multiple perturbation types. Defaults to False.
            cpd_name (str, optional): Column name for compound names. Defaults to "CPD_NAME".
        """
        super(CellDatasetFold, self).__init__()

        self.image_path = image_path
        self.fold = fold  
        self.data = data
        self.dataset_name = dataset_name
        self.add_controls = add_controls
        self.batch_correction = batch_correction
        self.multimodal = multimodal
        self.cpd_name = cpd_name
        
        # Extract variables
        if self.batch_correction:
            self.file_names = data['SAMPLE_KEY']
            self.mols = data[batch_key]
            self.y = data['ANNOT']
            if dataset_name == "bbbc021":
                self.dose = data['DOSE']
        else:
            self.file_names = {}
            self.mols = {}
            self.y = {}
            if dataset_name == "bbbc021":
                self.dose = {}
            
            for cond in ["ctrl", "trt"]:
                if cond == "trt" and add_controls:
                    self.file_names[cond] = self.data['SAMPLE_KEY']
                    self.mols[cond] = self.data['CPD_NAME']
                    self.y[cond] = self.data['ANNOT']
                    if dataset_name == "bbbc021":
                        self.dose[cond] = self.data['DOSE']
                else:
                    self.file_names[cond] = self.data['SAMPLE_KEY'][self.data[f"{cond}_idx"]]
                    self.mols[cond] = self.data['CPD_NAME'][self.data[f"{cond}_idx"]]
                    self.y[cond] = self.data['ANNOT'][self.data[f"{cond}_idx"]]
                    if dataset_name == "bbbc021":
                        self.dose[cond] = self.data['DOSE'][self.data[f"{cond}_idx"]]
                        
        del data 
        
        # Whether to perform training augmentation
        self.augment_train = augment_train
        
        # One-hot encoders 
        self.mol2id = mol2id
        self.y2id = y2id
        
        # Transform only the training set and only if required
        self.transform = CustomTransform(augment=(self.augment_train and self.fold == 'train'), normalize=normalize)
        
    def __len__(self):
        """
        Return the total number of samples.
        
        Returns:
            int: Number of samples.
        """
        if self.batch_correction:
            return len(self.file_names)
        else:
            return len(self.file_names["ctrl"])

    def __getitem__(self, idx):
        """
        Generate one example datapoint.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            dict: Dictionary containing the image tensor, one-hot encoded molecule, annotation ID, dose, and file name.
        """
        # Image must be fetched from disk
        if self.batch_correction:
            return read_files_batch(self.file_names, 
                                    self.mols,
                                    self.mol2id,
                                    self.y2id, 
                                    self.y, 
                                    self.transform,
                                    self.image_path, 
                                    self.dataset_name, 
                                    idx)
        else:
            return read_files_pert(self.file_names, 
                                   self.mols, 
                                   self.mol2id, 
                                   self.y2id, 
                                   self.dose, 
                                   self.y, 
                                   self.transform, 
                                   self.image_path, 
                                   self.dataset_name,
                                   idx,
                                   self.multimodal)

class CellDataLoader(LightningDataModule):
    """
    General data loader class for PyTorch Lightning.

    This class handles the creation of data loaders for training and testing, 
    including the initialization of datasets and batch processing.
    """
    
    def __init__(self, args):
        """
        Initialize the CellDataLoader instance.
        
        Args:
            args (argparse.Namespace): Arguments containing dataloader configuration.
        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.init_dataset()
        
    def create_torch_datasets(self):
        """
        Create datasets compatible with the PyTorch training loop.
        
        Returns:
            tuple: Training and test datasets.
        """
        dataset = CellDataset(self.args, device=self.device) 
        
        # Channel dimension
        self.dim = self.args.n_channels

        # Integrate embeddings as class attribute
        self.embedding_matrix = dataset.embedding_matrix  

        # Number of molecules and annotations (the latter can be modes of action/genes...)
        self.n_mol = dataset.n_mol
        self.num_y = dataset.n_y 

        # Collect training and test set
        training_set, test_set = dataset.fold_datasets.values()  
        
        # Collect IDs
        self.mol2id = dataset.mol2id
        self.y2id = dataset.y2id
        self.id2mol = {val: key for key, val in self.mol2id.items()}
        self.id2y = {val: key for key, val in self.y2id.items()}   

        # Free cell painting dataset memory
        del dataset
        return training_set, test_set
        
    def init_dataset(self):
        """
        Initialize dataset and data loaders.
        """
        self.training_set, self.test_set = self.create_torch_datasets()
        
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
        
    
    def train_dataloader(self):
        """
        Return the training data loader.
        
        Returns:
            DataLoader: Training data loader.
        """
        return self.loader_train
    
    def val_dataloader(self):
        """
        Return the validation data loader.
        
        Returns:
            DataLoader: Validation data loader.
        """
        return self.loader_test
    
    def test_dataloader(self):
        """
        Return the test data loader.
        
        Returns:
            DataLoader: Test data loader.
        """
        return self.loader_test

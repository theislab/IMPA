import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset

from pytorch_lightning import LightningDataModule
from IMPA.dataset.data_utils import CustomTransform
from IMPA.utils import *
from tqdm import tqdm

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
        self.add_controls = args.add_controls
        self.multimodal = args.multimodal

        # Fix the training specifics 
        self.device = device 

        # Read the datasets
        self.fold_datasets = self._read_folds()
        
        # ANNOT HERE IS FUNDAMENTAL BEACUSE IT CONTROLS WHAT TYPE OF PERT IT IS 
        self.y_names = np.unique(self.fold_datasets['train']["ANNOT"])  # ORF, Compound, CRISPR
        if not self.multimodal:
            # Count the number of compounds
            if args.add_controls:
                # BROAD SAMPLES FOR ALL (ALSO DMSO OR CONTROLS)
                self.mol_names = np.unique(self.fold_datasets["train"][self.cpd_name])
            else:
                # BROAD SAMPLES ONLY FOR THE TREATED (BROAD ID FOR ALL)
                self.mol_names = np.unique(self.fold_datasets["train"][self.cpd_name][self.fold_datasets["train"]["trt_idx"]]) # Sorted drug names
            self.n_mol = len(self.mol_names) 

        else:
            self.mol_names = {}
            for pert_type in self.y_names:
                idx_pert = self.fold_datasets["train"]["ANNOT"]==pert_type
                if args.add_controls:
                    # BROAD SAMPLES FOR ALL (ALSO DMSO OR CONTROLS)
                    self.mol_names[pert_type] = np.unique(self.fold_datasets["train"][self.cpd_name][idx_pert])
                else:
                    # BROAD SAMPLES ONLY FOR THE TREATED (BROAD ID FOR ALL)
                    trt_idx = self.fold_datasets["train"]["trt_idx"][idx_pert]
                    self.mol_names[pert_type] = np.unique(self.fold_datasets["train"][self.cpd_name][idx_pert][trt_idx])
            self.n_mol = {key: len(val) for key, val in self.mol_names.items()} 

        # Count the number of drugs and MOAs 
        self.n_y = len(self.y_names)
        self.y2id = {y: id for id, y in enumerate(self.y_names)}

        # Initialize the embedding matrix
        if self.multimodal and not self.trainable_emb:
            embedding_matrix = []
            mol2id = {}
            encoder_mol = {}
            self.latent_dim = {}

            for mod in self.y_names:
                embedding_matrix_modality = pd.read_csv(self.embedding_path[mod],
                                                        index_col=0)
                embedding_matrix_modality = embedding_matrix_modality.loc[self.mol_names[mod]]
                embedding_matrix_modality = torch.tensor(embedding_matrix_modality.values, 
                                                            dtype=torch.float32, device=self.device)
                self.latent_dim[mod]= embedding_matrix_modality.shape[1]
                embedding_matrix_modality = torch.nn.Embedding.from_pretrained(embedding_matrix_modality, 
                                                                                    freeze=True).to(self.device)
                embedding_matrix.append(embedding_matrix_modality)
                mol2id[mod] = {mol: id for id, mol in enumerate(self.mol_names[mod])}
                    
                # Encoders for moa and drug 
                encoder_mol[mod] = OneHotEncoder(sparse=False, categories=[self.mol_names[mod]])
                encoder_mol[mod].fit(np.array(self.mol_names[mod]).reshape((-1,1)))
                
            self.embedding_matrix = torch.nn.ModuleList(embedding_matrix)
            self.mol2id = mol2id

        else:
            if self.trainable_emb:
                self.embedding_matrix = torch.nn.Embedding(self.n_mol, self.latent_dim).to(self.device).to(torch.float32) 
            else:
                embedding_matrix = pd.read_csv(self.embedding_path, index_col=0)
                embedding_matrix = embedding_matrix.loc[self.mol_names]  # Sort based on the drug names 
                embedding_matrix = torch.tensor(embedding_matrix.values, 
                                                dtype=torch.float32, device=self.device)
                self.embedding_matrix = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=True).to(self.device)

            # Keep track of the indices and numbers 
            self.mol2id = {mol: id for id, mol in enumerate(self.mol_names)}
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
                                                        self.normalize, 
                                                        dataset_name=self.dataset_name, 
                                                        add_controls=self.add_controls,
                                                        multimodal=self.multimodal, 
                                                        cpd_name=self.cpd_name),


                                'test': CellDatasetFold('test', 
                                                        self.image_path,
                                                        self.fold_datasets['test'], 
                                                        encoder_mol, 
                                                        self.mol2id, 
                                                        self.y2id, 
                                                        self.augment_train, 
                                                        self.normalize, 
                                                        dataset_name=self.dataset_name,
                                                        add_controls=self.add_controls,
                                                        multimodal=self.multimodal, 
                                                        cpd_name=self.cpd_name)}  

    def _read_folds(self):
        """Extract the filenames of images in the train and test sets 
        associated folder
        """
        # Read the index csv file
        dataset = pd.read_csv(self.data_index_path, index_col=0)

        self.cpd_name = "BROAD_SAMPLE" if self.dataset_name=="cpg0000" else "CPD_NAME"
        # Subset the perturbations if provided in mol_list
        if self.mol_list != None:
            dataset = dataset.loc[dataset[self.cpd_name].isin(self.mol_list)]
        # Remove the leave-out drugs if provided in ood_set
        if self.ood_set != None:
            dataset = dataset.loc[~dataset[self.cpd_name].isin(self.ood_set)]
        
        # Collect in a dictionary the folds
        dataset_splits = dict()
        
        for fold_name in ['train', 'test']:
            # Divide the dataset in splits 
            dataset_splits[fold_name] = {}

            # Subset of dataframe corresponding to the split 
            subset = dataset.loc[dataset.SPLIT == fold_name]
            # Save the fold name 
            for key in subset.columns:
                dataset_splits[fold_name][key] = np.array(subset[key])
            if not self.add_controls:
                dataset_splits[fold_name]["trt_idx"] = (dataset_splits[fold_name]["STATE"]=="trt")
            else:
                dataset_splits[fold_name]["trt_idx"] = (np.isin(dataset_splits[fold_name]["STATE"], ["trt", "control"]))
            dataset_splits[fold_name]["ctrl_idx"] = (dataset_splits[fold_name]["STATE"]=="control")
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
                normalize=False, 
                dataset_name="bbbc021", 
                add_controls=False, 
                multimodal=False, 
                cpd_name="CPD_NAME"):

        super(CellDatasetFold, self).__init__() 

        # Train or test sets
        self.image_path = image_path
        self.fold = fold  
        self.data = data
        self.dataset_name = dataset_name
        self.multimodal = multimodal
        
        # Extract variables 
        self.file_names = {}
        self.mols = {}
        self.y = {}
        
        if dataset_name=="bbbc021":
            self.dose = {}
        
        for cond in ["ctrl", "trt"]:
            # subset by condition 
            if cond == "trt" and add_controls:
                self.file_names[cond] = self.data['SAMPLE_KEY']
                self.mols[cond] = self.data[cpd_name]
                self.y[cond] = self.data['ANNOT']
                if dataset_name=="bbbc021":
                    self.dose[cond] = self.data['DOSE']
                    
            else:
                self.file_names[cond] = self.data['SAMPLE_KEY'][self.data[f"{cond}_idx"]]
                self.mols[cond] = self.data[cpd_name][self.data[f"{cond}_idx"]]
                self.y[cond] = self.data['ANNOT'][self.data[f"{cond}_idx"]]
                if dataset_name=="bbbc021":
                    self.dose[cond] = self.data['DOSE'][self.data[f"{cond}_idx"]]
         
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
                    
    def __len__(self):
        """
        Total number of samples 
        """
        return len(self.file_names["ctrl"])

    def __getitem__(self, i):
        """
        Generate one example datapoint 
        """
        # Sample control and treated batches 
        img_file_ctrl = self.file_names["ctrl"][i]
        idx_trt = np.random.randint(0, len(self.file_names["trt"]))
        img_file_trt = self.file_names["trt"][idx_trt]
    
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
            if self.dataset_name=="cpg0000":
                path_ctrl = Path(self.image_path) / file_split_ctrl[0] / f"{file_split_ctrl[1]}_{file_split_ctrl[2]}"
                path_trt = Path(self.image_path) / file_split_trt[0] / f"{file_split_trt[1]}_{file_split_trt[2]}"
                file_ctrl = '_'.join(file_split_ctrl[1:])+".npy"
                file_trt = '_'.join(file_split_trt[1:])+".npy"
            elif self.dataset_name=="bbbc021":
                path_ctrl = Path(self.image_path) / file_split_ctrl[0] / f"{file_split_ctrl[1]}"
                path_trt = Path(self.image_path) / file_split_trt[0] / f"{file_split_trt[1]}"
                file_ctrl = '_'.join(file_split_ctrl[2:])+".npy"
                file_trt = '_'.join(file_split_trt[2:])+".npy"
            
        img_ctrl, img_trt = np.load(path_ctrl / file_ctrl), np.load(path_trt / file_trt)
        img_ctrl, img_trt = torch.from_numpy(img_ctrl).to(torch.float), torch.from_numpy(img_trt).to(torch.float)
        img_ctrl, img_trt = img_ctrl.permute(2,0,1), img_trt.permute(2,0,1)  # Place channel dimension in front of the others 
        img_ctrl, img_trt = self.transform(img_ctrl), self.transform(img_trt)
        if self.multimodal:
            y_mod = self.y["trt"][idx_trt]
            mol_one_hot = self.mol2id[y_mod][self.mols["trt"][idx_trt]]
        else:
            mol_one_hot = self.mol2id[self.mols["trt"][idx_trt]]
        
        if self.dataset_name == "bbbc021":
            return {'X':(img_ctrl, img_trt), 
                    'mol_one_hot': mol_one_hot, 
                    'y_id': self.y2id[self.y["trt"][idx_trt]],
                    "dose": self.dose["trt"][idx_trt],
                    'file_names': (img_file_ctrl, img_file_trt)}
        else:
            return {'X':(img_ctrl, img_trt), 
                    'mol_one_hot': mol_one_hot, 
                    'y_id': self.y2id[self.y["trt"][idx_trt]],
                    'file_names': (img_file_ctrl, img_file_trt)}

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
        self.latent_dim = dataset.latent_dim
        print(self.latent_dim)

        # Number of mols and annotations (the latter can be modes of action/genes...)
        self.n_mol = dataset.n_mol
        self.num_y = dataset.n_y 

        # Collect training and test set 
        training_set, test_set = dataset.fold_datasets.values() 
        
        # Collect ids 
        self.mol2id = dataset.mol2id
        self.y2id = dataset.y2id
        if self.args.multimodal:
            self.id2mol = {}
            self.id2y = {}
            for mod in self.mol2id:
                self.id2mol[mod] = {val:key for key,val in self.mol2id[mod].items()}
                self.id2y[mod] = {val:key for key,val in self.y2id.items()} 
        else:
            self.id2mol = {val:key for key,val in self.mol2id.items()}
            self.id2y = {val:key for key,val in self.y2id.items()} 
            
        # Free cell painting dataset memory
        del dataset
        return training_set, test_set
        
        
    def init_dataset(self):
        """Initialize dataset and data loaders
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
        return self.loader_train
    
    def val_dataloader(self):
        return self.loader_test
    
    def val_dataloader(self):
        return self.loader_test
    
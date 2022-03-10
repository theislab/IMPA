"""
Implement custom dataset 
"""
import numpy as np
from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
from compert.utils import *
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import os


class CustomTransform:
    """
    Scale and resize an input image 
    """
    def __init__(self, augment = False):
        self.augment = augment 
        
    def __call__(self, X):
        """
        Transform the image at the time of loading by rescaling and turning into a tensor. (NB images already normalizes) 
        """
        # Add random noise and rescale pixels between 0 and 1 
        random_noise = torch.rand_like(X)
        X = (X+random_noise)/255.0
        # Compute means and std per channels
        if self.augment:
        # Apply resize 
            transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(np.random.randint(0,30)),
                T.RandomVerticalFlip(p=0.5)]
            )
            return transform(X)
        else:
            return X



class CellPaintingDataset:
    """
    Dataset class for image data 
    """
    def __init__(self, image_path, data_index_path, embeddings_path, device='cuda', return_labels=True, augment_train=False, use_pretrained=True):
        """
        Params:
        -------------
            :data_path: the repository where the data is stored 
            :transform: a pytorch transform object to apply augmentation to the data 
            :data_index_path: path to .npz object with sample names, molecule names, molecule smiles and the assay labels
            :return_labels: bool to assess whether to return labels together with observations in __getitem__
        """    
        assert os.path.exists(image_path), 'The data path does not exist'
        assert os.path.exists(data_index_path), 'The data index path does not exist'

        # Set up the variables 
        self.image_path = image_path 
        self.data_index_path = data_index_path
        self.embeddings_path = embeddings_path
        self.device = device 
        self.return_labels = return_labels  # Whether onehot drug labels, the assay labels and the case/control labels should be returned 
        self.use_pretrained = use_pretrained

        # Read the datasets
        self.fold_datasets = self.read_folds()
        
        # Take the seen molecules from the training, test and valid set and map them to indices 
        self.seen_compounds = np.sort(np.unique(self.fold_datasets['train']['mol_names']))
        self.n_seen_drugs = len(self.seen_compounds)

        # Onehot encoder 
        encoder_drug = OneHotEncoder(sparse=False, categories=[self.seen_compounds])
        encoder_drug.fit(np.array(self.seen_compounds).reshape((-1,1)))

        # Prepare the embeddings     
        self.all_drugs = np.sort(list(set(self.fold_datasets['train']['mol_smiles']).union(set(self.fold_datasets['ood']['mol_smiles'])).
                                    union(set(self.fold_datasets['test']['mol_smiles'])).union(set(self.fold_datasets['val']['mol_smiles']))))
                                        
        self.num_drugs = len(self.all_drugs)

        # Each smile has an id
        drug_embeddings = pd.read_csv(self.embeddings_path, index_col=0).loc[self.all_drugs]  # Read embedding paths
        # drug_embeddings = drug_embeddings.sort_index()
        # all_smiles = drug_embeddings.index
        if self.use_pretrained:
            # Tranform the embddings to torch embeddings
            drug_embeddings  = torch.tensor(drug_embeddings.loc[self.all_drugs].values, 
                                                dtype=torch.float32, device=self.device)
            self.drug_embeddings = torch.nn.Embedding.from_pretrained(drug_embeddings, freeze=True).to(self.device)
    
        mol2label = {d:i for d,i in zip(self.all_drugs, range(len(self.all_drugs)))}  

        # Initialize the datasets 
        self.fold_datasets = {'train': CellPaintingFold('train', self.fold_datasets['train'], 
                                                        self.image_path, 
                                                        encoder_drug, 
                                                        mol2label, 
                                                        self.return_labels, 
                                                        use_pretrained,
                                                        augment_train),
                            'val': CellPaintingFold('val', self.fold_datasets['val'], 
                                                    self.image_path, 
                                                    encoder_drug, 
                                                    mol2label, 
                                                    self.return_labels,
                                                    use_pretrained),
                            'test': CellPaintingFold('test', self.fold_datasets['test'], 
                                                        self.image_path, 
                                                        encoder_drug, 
                                                        mol2label, 
                                                        self.return_labels,
                                                        use_pretrained),
                            'ood': CellPaintingFold('ood', self.fold_datasets['ood'], 
                                                    self.image_path, 
                                                    encoder_drug, 
                                                    mol2label, 
                                                    self.return_labels,
                                                    use_pretrained)}
        
        
    def read_folds(self):
        """
        Extract the filenames of images in the train, test and validation sets from the 
        associated folder
        """
        # Get the file names and molecules of training, test and validation sets
        datasets = dict()
        for fold_name in ['train', 'val', 'test', 'ood']:
            datasets[fold_name] = {}
            # Fetch the data
            data_index_path = os.path.join(self.data_index_path, f'{fold_name}_data_index.npz')
            # Get the files with the sample splits and add them to the dictionary 
            fold_file, mol_names, mol_smiles, assay_labels, states  = self.get_files_and_mols_from_path(data_index_path=data_index_path)

            # Add the  important entries to the dataset
            datasets[fold_name]['file_names'] = fold_file
            datasets[fold_name]['mol_names'] = mol_names
            datasets[fold_name]['mol_smiles'] = mol_smiles
            datasets[fold_name]['assay_labels'] = assay_labels
            datasets[fold_name]['state'] = states
        return datasets
    
    def get_files_and_mols_from_path(self, data_index_path): 
        """
        Load object with image names, molecule names and smiles 
        -------------------
        data_index_path: The path to the data index with information about molecules and sample names
        """
        assert os.path.exists(data_index_path), 'The data index file does not exist'
        # Load the index file 
        file = np.load(data_index_path, allow_pickle=True)

        file_names = file['filenames']
        mol_names = file['mol_names']
        mol_smiles =  file['mol_smiles']
        assay_labels = file['assay_labels']
        if 'state' in file:
            states = file['state']
            states = np.where(states=='ctr', 0., 1.)
        else:
            states = None 
        # Convert state to binary
        return file_names, mol_names, mol_smiles, assay_labels, states
    

class CellPaintingFold(Dataset):
    def __init__(self, fold, data, image_path, drug_encoder, mol2label, return_labels = True, use_pretrained=True, augment_train=True):
        super(CellPaintingFold, self).__init__() 
        
        self.fold = fold
        # For each piece of the data create its own object
        self.file_names = data['file_names']
        self.mol_names = data['mol_names']
        self.mol_smiles = data['mol_smiles']
        self.assay_labels = data['assay_labels']
        self.states = data['state']

        # Data paths 
        self.data_path = image_path
        # Whether to perform training augmentation
        self.augment_train = augment_train
        
        # Drug data
        self.drug_encoder = drug_encoder
        self.mol2label = mol2label
        
        if self.augment_train and self.fold == 'train':
            self.transform = CustomTransform(augment=True)
        else:
            self.transform = CustomTransform(augment=False)

        # Control whether the labels should be provided in the batch together with the images  
        self.return_labels = return_labels 
        self.use_pretrained = use_pretrained

        # One -hot encode molecules
        if self.fold != 'ood':
            self.one_hot_drugs = self.drug_encoder.transform(np.array(self.mol_names.reshape((-1,1))))
        
        
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
        sample = img_file.split('-')[0]
        well = img_file.split('-')[1].split('-')[0]
        # Load image 
        if self.states[idx] == 1:
            cond = 'trt_images'
        else:
            cond = 'ctr_images'
        with np.load(os.path.join(self.data_path, cond,  sample, well, f'{img_file}.npz'), allow_pickle = True) as f:
            img = f['arr_0']
        img = torch.from_numpy(img).to(torch.float)
        img = img.permute(2,0,1)  # Place channel dimension in front of the others 
        if self.transform != None:
            img = self.transform(img)

        if self.return_labels:
            return dict(X=img, 
                        file_name=img_file,
                        mol_name=self.mol_names[idx], 
                        mol_one_hot=self.one_hot_drugs[idx] if self.fold != 'ood' else '',
                        assay_labels=self.assay_labels[idx],
                        state = self.states[idx],
                        smile_id = self.mol2label[self.mol_smiles[idx]])
        else:
            return dict(X=img)

    def sample(self, n, seed=42):
        """
        Sample random observations from the training set
        """
        np.random.seed(seed)
        # Pick indices
        idx = np.arange(len(self.file_names))
        idx_sample = np.random.choice(idx, n, replace=False)

        # Select from the the filenames at random
        imgs = []
        subset_file_names = []
        subset_mol_name = []
        subset_mol_one_hot = []
        subset_assay_labels = []
        subset_states = []
        subset_smile_id = []
        
        
        for i in idx_sample:
            X, file_name, mol_name, mol_one_hot, assay_label, states, smile_id = self.__getitem__(i).values()
            imgs.append(X.unsqueeze(0))
            subset_file_names.append(file_name)
            subset_mol_name.append(mol_name)
            subset_mol_one_hot.append(mol_one_hot)
            subset_assay_labels.append(assay_label)
            subset_states.append(states)
            subset_smile_id.append(smile_id)

        imgs = torch.cat(imgs, dim=0)
        return dict(X=imgs, 
                    file_name=subset_file_names,
                    mol_name=subset_mol_name, 
                    mol_one_hot=self.one_hot_drugs[idx] if self.fold != 'ood' else '',
                    assay_labels=subset_assay_labels,
                    state = subset_states,
                    smile_id = subset_smile_id) 


        

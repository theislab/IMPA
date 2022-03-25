"""
Implement custom dataset 
"""
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

from utils import *

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CustomTransform:
    """
    Scale and resize an input image 
    """
    def __init__(self, augment = False):
        self.augment = augment 
        
    def __call__(self, X):
        """
        Transform the image at the time of loading by rescaling and turning into a tensor
        """
        # Add random noise and rescale pixels between 0 and 1 
        random_noise = torch.rand_like(X)
        X = (X+random_noise)/255.0
        # Perform augmentation step
        if self.augment:
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
    def __init__(self, image_path, data_index_path, embeddings_path, device='cuda', return_labels=False, augment_train=False, use_pretrained=True):    
        assert os.path.exists(image_path), 'The data path does not exist'
        assert os.path.exists(data_index_path), 'The data index path does not exist'

        # Set up the variables 
        self.image_path = image_path  # Path to the images 
        self.data_index_path = data_index_path  # Path to data index (.npy object storing the splits) 
        self.embeddings_path = embeddings_path  
        self.device = device 
        self.return_labels = return_labels  # Whether onehot drug labels, the assay labels and the case/control labels should be returned 
        self.use_pretrained = use_pretrained

        # Read the datasets
        self.fold_datasets = self.read_folds()
        
        # Take the seen molecules from the training, test and valid set and map them to indices 
        self.seen_compounds = np.sort(np.unique(self.fold_datasets['train']['mol_names']))  # Training set contains all the seen molecules 
        self.n_seen_drugs = len(self.seen_compounds)

        # Onehot encoder for the seen compounds
        encoder_drug = OneHotEncoder(sparse=False, categories=[self.seen_compounds])
        encoder_drug.fit(np.array(self.seen_compounds).reshape((-1,1)))

        # Prepare the embeddings     
        self.all_drugs = np.sort(list(set(self.fold_datasets['train']['mol_smiles']).union(set(self.fold_datasets['ood']['mol_smiles'])).
                                    union(set(self.fold_datasets['test']['mol_smiles'])).union(set(self.fold_datasets['val']['mol_smiles']))))                                        
        self.num_drugs = len(self.all_drugs)  # Total number of drugs 

        if self.use_pretrained:
            # Each smile has an id
            drug_embeddings = pd.read_csv(self.embeddings_path, index_col=0).loc[self.all_drugs]  # Read embedding paths
            # Tranform the embddings to torch embeddings
            drug_embeddings  = torch.tensor(drug_embeddings.values, 
                                                dtype=torch.float32, device=self.device)
            # Must feed from_pretrained() with num_embeddings x dimension
            self.drug_embeddings = torch.nn.Embedding.from_pretrained(drug_embeddings, freeze=True).to(self.device)
    
        # Dictionary with every drug associated to an id 
        mol2label = {d:i for i,d in enumerate(self.all_drugs)}  


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
            fold_file, mol_names, mol_smiles, assay_labels, states, n_cells  = self.get_files_and_mols_from_path(data_index_path=data_index_path)

            # Add the  important entries to the dataset
            datasets[fold_name]['file_names'] = fold_file
            datasets[fold_name]['mol_names'] = mol_names
            datasets[fold_name]['mol_smiles'] = mol_smiles
            datasets[fold_name]['assay_labels'] = assay_labels
            datasets[fold_name]['state'] = states
            datasets[fold_name]['n_cells'] = n_cells 
        return datasets

    
    def get_files_and_mols_from_path(self, data_index_path): 
        """Retrieve the data from within a data index npy object 

        Args:
            data_index_path (str): The path to the data index of interest 
        """
        assert os.path.exists(data_index_path), 'The data index file does not exist'
        # Load the index file 
        file = np.load(data_index_path, allow_pickle=True)

        file_names = file['filenames']
        mol_names = file['mol_names']
        mol_smiles =  file['mol_smiles']
        assay_labels = file['assay_labels']
        n_cells = file['n_cells']
        # State contains 1 or 0 labels representing inactive vs active compounds  
        if 'state' in file:
            # Convert state to binary
            states = file['state']
            states = np.where(states=='ctr', 0., 1.)
        else:
            states = None 
        return file_names, mol_names, mol_smiles, assay_labels, states, n_cells
    

class CellPaintingFold(Dataset):
    def __init__(self, fold, data, image_path, drug_encoder, mol2label, return_labels = True, use_pretrained=True, augment_train=True):
        super(CellPaintingFold, self).__init__() 
        
        self.fold = fold  # train, test or validation set
        # For each piece of the data create its own object
        self.file_names = data['file_names']
        self.mol_names = data['mol_names']
        self.mol_smiles = data['mol_smiles']
        #self.assay_labels = data['assay_labels']
        self.states = data['state']
        self.n_cells = data['n_cells']

        # Data paths 
        self.data_path = image_path
        # Whether to perform training augmentation
        self.augment_train = augment_train
        
        # Drug data
        self.drug_encoder = drug_encoder

        # Subset mol2label to the important part only 
        self.mol2label = {key:value for key, value in mol2label.items() if key in self.mol_smiles}
        
        # Transform only the training set and only if required
        if self.augment_train and self.fold == 'train':
            self.transform = CustomTransform(augment=True)
        else:
            self.transform = CustomTransform(augment=False)

        # Control whether the labels should be provided in the batch together with the images  
        self.return_labels = return_labels 
        self.use_pretrained = use_pretrained

        # One-hot encode molecules
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
        # Image must be fetched from disk 
        img_file = self.file_names[idx]
        sample = img_file.split('-')[0]
        well = img_file.split('-')[1].split('-')[0]
        # Load image from one of the folders between trt and ctr
        if self.states[idx] == 1:
            cond = 'trt_images'
        else:
            cond = 'ctr_images'
        with np.load(os.path.join(self.data_path, cond, sample, well, f'{img_file}.npz'), allow_pickle = True) as f:
            img = f['arr_0']
        img = torch.from_numpy(img).to(torch.float)
        img = img.permute(2,0,1)  # Place channel dimension in front of the others 
        img = self.transform(img)

        if self.return_labels:
            return dict(X=img, 
                        #file_name=img_file,
                        #mol_name=self.mol_names[idx], 
                        mol_one_hot=self.one_hot_drugs[idx] if self.fold != 'ood' else '',
                        #assay_labels=self.assay_labels[idx],
                        state = self.states[idx],
                        smile_id = self.mol2label[self.mol_smiles[idx]],
                        n_cells = self.n_cells[idx])
        else:
            return dict(X=img)


    def sample(self, n, seed=42):
        """Sample a batch of random observations

        Args:
            n (int): Number of sampled observations
            seed (int, optional): The random seed for reproducibility. Defaults to 42.

        Returns:
            dict: Dictionary containing a random data batch.
        """
        np.random.seed(seed)
        # Pick indices
        idx = np.arange(len(self.file_names))
        idx_sample = np.random.choice(idx, n, replace=False)

        # Select from the the filenames at random
        imgs = []
        subset_mol_one_hot = []
        subset_states = []
        subset_smile_id = []
        subset_n_cells = []
        
        
        for i in idx_sample:
            #X, file_name, mol_name, mol_one_hot, assay_label, states, smile_id, n_cells = self.__getitem__(i).values()
            X, mol_one_hot, states, smile_id, n_cells = self.__getitem__(i).values()
            imgs.append(X.unsqueeze(0))  # Unsqueeze barch dimension
            subset_mol_one_hot.append(mol_one_hot)
            subset_states.append(states)
            subset_smile_id.append(smile_id)
            subset_n_cells.append(n_cells)

        imgs = torch.cat(imgs, dim=0)
        return dict(X=imgs,
                    mol_one_hot=self.one_hot_drugs[idx] if self.fold != 'ood' else '',
                    state = subset_states,
                    smile_id = subset_smile_id, 
                    n_cells = subset_n_cells) 

    
    def get_drug_by_name(self, drug_name):
        """Load the images of specific drugs by name

        Args:
            drug_name (str): The name of the drug of interest  
        """
        # Get the indexes in the specific set of all the drugs with a specific name
        drug_idxs = [idx for idx in range(len(self.mol_names)) if self.mol_names[idx]==drug_name]

        # Collect the drug images from the file repository 
        drug_images = []
        for drug_idx in drug_idxs:
            #X, _, mol_name, mol_one_hot, assay_label, states, smile_id, n_cells = self.__getitem__(drug_idx).values()
            X, mol_one_hot, states, smile_id, n_cells = self.__getitem__(drug_idx).values()
            drug_images.append(X.unsqueeze(0))

        return dict(X=torch.cat(drug_images, dim=0), 
                mol_one_hot=mol_one_hot if self.fold != 'ood' else '',
                state = states,
                smile_id = smile_id, 
                n_cells = n_cells) 


        

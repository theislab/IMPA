"""
Implement custom dataset 
"""

from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
from utils import *
from sklearn.preprocessing import OneHotEncoder

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
    def __init__(self, data_path, device='cuda', return_labels=False, augment_train=False):
        """
        Params:
        -------------
            :data_path: the repository where the data is stored 
            :transform: a pytorch transform object to apply augmentation to the data 
            :data_index_path: path to .npz object with sample names, molecule names, molecule smiles and the assay labels
            :return_labels: bool to assess whether to return labels together with observations in __getitem__
        """    
        assert os.path.exists(data_path), 'The data path does not exist'

        # Read train, validation and test sets 
        self.data_path = data_path 
        self.device = device 
        self.return_labels = return_labels

        # Read the drug names
        print('Load the data')
        self.fold_datasets = self.read_folds()
        
        # Take the seen molecules from the training, test and valid set and map them to indices 
        seen_compounds = np.unique(self.fold_datasets['train']['mol_names'])
        unseen_compounds = np.unique(self.fold_datasets['ood']['mol_names'])
        assert len(seen_compounds)+len(unseen_compounds) == 10560
        
        seen_compounds = sorted(seen_compounds)
        mol2label = {d:i for d,i in zip(seen_compounds, range(len(seen_compounds)))}
        
        # Onehot encoder 
        encoder_drug = OneHotEncoder(sparse=False, categories=[seen_compounds])
        encoder_drug.fit(np.array(seen_compounds).reshape((-1,1)))
        
        # Initialize the datasets 
        self.fold_datasets = {'train': CellPaintingFold('train', self.fold_datasets['train'], self.data_path, encoder_drug, mol2label, self.return_labels, augment_train),
                         'val': CellPaintingFold('val', self.fold_datasets['val'], self.data_path, encoder_drug, mol2label, self.return_labels),
                         'test': CellPaintingFold('test', self.fold_datasets['test'], self.data_path, encoder_drug, mol2label, self.return_labels),
                         'ood': CellPaintingFold('ood', self.fold_datasets['ood'], self.data_path, encoder_drug, mol2label, self.return_labels)}
        
        
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
            data_index_path = os.path.join(self.data_path, f'{fold_name}_data_index.npz')
            # Get the files with the sample splits and add them to the dictionary 
            fold_file, mol_names, mol_smiles, assay_labels  = self.get_files_and_mols_from_path(data_index_path=data_index_path)

            # Add the  important entries to the dataset
            datasets[fold_name]['file_names'] = fold_file
            datasets[fold_name]['mol_names'] = mol_names
            datasets[fold_name]['mol_smiles'] = mol_smiles
            datasets[fold_name]['assay_labels'] = assay_labels
        return datasets
    
    def get_files_and_mols_from_path(self, data_index_path): 
        """
        Load object with image names, molecule names and smiles 
        -------------------
        data_index_path: The path to the data index with information about molecules and sample names
        """
        assert os.path.exists(data_index_path), 'The data index file does not exist'
        # Load the index file 
        file = np.load(data_index_path, allow_pickle= True)

        file_names = file['filenames']
        mol_names = file['mol_names']
        mol_smiles =  file['mol_smiles']
        assay_labels = file['assay_labels']
        return file_names, mol_names, mol_smiles, assay_labels
    

class CellPaintingFold(Dataset):
    def __init__(self, fold, data, data_path, drug_encoder, mol2label, return_labels = True, augment_train=True):
        super(CellPaintingFold, self).__init__() 
        
        self.fold = fold
        # For each piece of the data create its own object
        self.file_names = data['file_names']
        self.mol_names = data['mol_names']
        self.mol_smiles = data['mol_smiles']
        self.assay_labels = data['assay_labels']
        self.data_path = data_path
        self.augment_train = augment_train
        
        self.drug_encoder = drug_encoder
        self.mol2label = mol2label
        
        if self.augment_train and self.fold == 'train':
            self.transform = CustomTransform(augment=True)
        else:
            self.transform = CustomTransform(augment=False)

        self.return_labels = return_labels 
        self.augment_train = augment_train
    
        # One -hot encode molecules
        if self.fold != 'ood':
            self.one_hot_drugs = self.drug_encoder.transform(np.array(self.mol_names.reshape((-1,1))))
        
        if self.augment_train and self.fold == 'train':
            self.transform.augment = True
        
        
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
        with np.load(os.path.join(self.data_path, sample, well, f'{img_file}.npz'), allow_pickle = True) as f:
            img = f['arr_0']
        img = torch.from_numpy(img).to(torch.float)
        img = img.permute(2,0,1)  # Place channel dimension in front of the others 
        if self.transform != None:
            img = self.transform(img)
        
        if self.fold == 'ood':
            if self.return_labels:
                return dict(X=img, 
                            file_name=img_file,
                            mol_name=self.mol_names[idx], 
                            mol_smile=self.mol_smiles[idx],
                            assay_labels=self.assay_labels[idx])
            else:
                return dict(X=img)
        else:        
            if self.return_labels:
                return dict(X=img, 
                            file_name=img_file,
                            mol_name=self.mol_names[idx], 
                            mol_one_hot=self.one_hot_drugs[idx],
                            mol_smile=self.mol_smiles[idx],
                            assay_labels=self.assay_labels[idx])
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
        subset_mol_one_hot = []
        subset_mol_smiles = []
        subset_assay_labels = []
        subset_file_names = []
        imgs = []
        
        if self.fols != 'ood':
            for i in idx_sample:
                X, file_name, mol_one_hot, mol_smile, assay_label = self.__getitem__(i).values()
                imgs.append(X.unsqueeze(0))
                subset_mol_one_hot.append(mol_one_hot)
                subset_mol_smiles.append(mol_smile)
                subset_assay_labels.append(assay_label)
                subset_file_names.append(file_name)

            imgs = torch.cat(imgs, dim=0)
            return dict(X=imgs, 
                        file_name=subset_file_names,
                        mols_one_hot=subset_mol_one_hot, 
                        mol_smile=subset_mol_smiles,
                        assay_label=subset_assay_labels) 

        else:
            for i in idx_sample:
                X, file_name, mol_smile, assay_label = self.__getitem__(i).values()
                imgs.append(X.unsqueeze(0))
                subset_mol_smiles.append(mol_smile)
                subset_assay_labels.append(assay_label)
                subset_file_names.append(file_name)
            
            imgs = torch.cat(imgs, dim=0)
            return dict(X=imgs, 
                        file_name=subset_file_names,
                        mol_smile=subset_mol_smiles,
                        assay_label=subset_assay_labels)
        

        

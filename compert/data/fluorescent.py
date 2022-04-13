"""
Implement custom dataset 
"""
import os
from random import sample
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
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
                T.RandomHorizontalFlip(p=0.3),
                T.RandomVerticalFlip(p=0.3)]
            )
            return transform(X)
        else:
            return X


class BBBC021Dataset:
    """
    Dataset class for image data 
    """
    def __init__(self, image_path, data_index_path, embeddings_path, device='cuda', return_labels=False, augment_train=False, use_pretrained=True):    
        assert os.path.exists(image_path), 'The data path does not exist'
        assert os.path.exists(data_index_path), 'The data index path does not exist'

        # Set up the variables 
        self.image_path = image_path  # Path to the image folder
        self.data_index_path = data_index_path  # Path to data index (.csv file) 
        self.embeddings_path = embeddings_path  # Path to the embedding matrix
        self.augment_train = augment_train

        # Fix the training specifics 
        self.device = device 
        self.return_labels = return_labels  # Whether onehot drug labels, the assay labels and the case/control labels should be returned 
        self.use_pretrained = use_pretrained  # Whether the pre-trained embeddings should be used for the drugs

        # Read the datasets
        self.fold_datasets = self.read_folds()
        self.drug2smile = {drug: smile for drug, smile in zip(self.fold_datasets['train']["mol_names"], self.fold_datasets['train']["mol_smiles"])}
        
        # Count the number of compounds 
        self.drug_names = np.sort(np.unique(self.fold_datasets['train']["mol_names"]))  # Sorted drug names
        self.moa_names = np.sort(np.unique(self.fold_datasets['train']["MOA"]))  # Sorted MOA names 
        self.smile_names = np.array([self.drug2smile[drug] for drug in self.drug_names])  # Sorted smile names based on drug sorting 

        # Count the number of drugs and MOAs 
        self.num_drugs = len(self.drug_names) 
        self.num_moa = len(self.moa_names)

        # Keep track of the indices and numbers 
        self.drugs2idx = {mol: idx for idx, mol in enumerate(self.drug_names)}
        self.moa2idx = {moa: idx for idx, moa in enumerate(self.moa_names)}

        # Encoders for moa and drug 
        encoder_drug = OneHotEncoder(sparse=False, categories=[self.drug_names])
        encoder_drug.fit(np.array(self.drug_names).reshape((-1,1)))

        encoder_moa = OneHotEncoder(sparse=False, categories=[self.moa_names])
        encoder_moa.fit(np.array(self.moa_names).reshape((-1,1)))

        if self.use_pretrained:
            # Get the drug embedding matrix indexed by indices 
            drug_embeddings = pd.read_csv(self.embeddings_path, index_col=0).loc[self.smile_names]  # Read embedding paths sorte based on smile names 
            # Tranform the embddings to torch embeddings
            drug_embeddings  = torch.tensor(drug_embeddings.values, 
                                                dtype=torch.float32, device=self.device)
            # Must feed from_pretrained() with num_embeddings x dimension
            self.drug_embeddings = torch.nn.Embedding.from_pretrained(drug_embeddings, freeze=True).to(self.device)

        
        # Initialize the datasets 
        self.fold_datasets = {'train': BBBC021Fold('train', self.fold_datasets['train'], 
                                                        self.image_path, 
                                                        encoder_drug, 
                                                        encoder_moa,
                                                        self.drugs2idx, 
                                                        self.moa2idx, 
                                                        self.return_labels, 
                                                        self.use_pretrained,
                                                        self.augment_train),

                            'val': BBBC021Fold('val', self.fold_datasets['valid'], 
                                                    self.image_path, 
                                                    encoder_drug, 
                                                    encoder_moa,
                                                    self.drugs2idx, 
                                                    self.moa2idx, 
                                                    self.return_labels, 
                                                    self.use_pretrained,
                                                    self.augment_train),

                            'test': BBBC021Fold('test', self.fold_datasets['test'], 
                                                    self.image_path, 
                                                    encoder_drug, 
                                                    encoder_moa,
                                                    self.drugs2idx, 
                                                    self.moa2idx, 
                                                    self.return_labels, 
                                                    self.use_pretrained,
                                                    self.augment_train),

                            'ood': BBBC021Fold('ood', self.fold_datasets['ood'], 
                                                    self.image_path, 
                                                    encoder_drug, 
                                                    encoder_moa,
                                                    self.drugs2idx, 
                                                    self.moa2idx, 
                                                    self.return_labels, 
                                                    self.use_pretrained,
                                                    self.augment_train)}



    def read_folds(self):
        """
        Extract the filenames of images in the train, test and validation sets from the 
        associated folder
        """
        # Read the index csv file
        dataset = pd.read_csv(os.path.join(self.data_index_path, 'dataset_splits.csv'))
        # Get the file names and molecules of training, test and validation sets
        dataset_splits = dict()
        
        for fold_name in ['train', 'valid', 'test', 'ood']:
            # Divide the dataset in splits 
            dataset_splits[fold_name] = {}

            # Subset of dataframe corresponding to the split 
            subset = dataset.loc[dataset.SPLIT == fold_name]

            # Add the  important entries to the dataset
            dataset_splits[fold_name]['file_names'] = np.array(subset.SAMPLE_KEY)
            dataset_splits[fold_name]['mol_names'] = np.array(subset.CPD_NAME)
            dataset_splits[fold_name]['mol_smiles'] = np.array(subset.SMILES)
            dataset_splits[fold_name]['dose'] = np.array(subset.DOSE)
            dataset_splits[fold_name]['MOA'] = np.array(subset.MOA)
        return dataset_splits



class BBBC021Fold(Dataset):
    def __init__(self, fold, data, image_path, drug_encoder, moa_encoder, drugs2idx, moa2idx, return_labels = True, use_pretrained=True, augment_train=True):
        super(BBBC021Fold, self).__init__() 
        
        self.fold = fold  # train, test or validation set
        # For each piece of the data create its own object
        self.file_names = data['file_names']
        self.mol_names = data['mol_names']
        self.mol_smiles = data['mol_smiles']
        self.dose = data['dose']
        self.moa = data['MOA']

        # Free memory due to the data
        del data 
        
        # Image folder paths 
        self.data_path = image_path
        # Whether to perform training augmentation
        self.augment_train = augment_train
        
        # One-hot encoders 
        self.drug_encoder = drug_encoder
        self.moa_encoder = moa_encoder

        # Subset mol2label to the important part only 
        self.drugs2idx = drugs2idx
        self.moa2idx = moa2idx
        
        # Transform only the training set and only if required
        if self.augment_train and self.fold == 'train':
            self.transform = CustomTransform(augment=True)
            # Compute class imbalance weights
            compute_class_imbalance_weights_drug = self.compute_class_imbalance_weights(self.mol_smiles)
            compute_class_imbalance_weights_moa = self.compute_class_imbalance_weights(self.moa)
            self.class_imbalances = {'drugs': compute_class_imbalance_weights_drug, 
                                'moas': compute_class_imbalance_weights_moa}
        else:
            self.transform = CustomTransform(augment=False)
        
        # Control whether the labels should be provided in the batch together with the images  
        self.return_labels = return_labels 
        self.use_pretrained = use_pretrained

        # One-hot encode molecules and moas
        self.one_hot_drugs = self.drug_encoder.transform(np.array(self.mol_names.reshape((-1,1))))
        self.one_hot_moa = self.moa_encoder.transform(np.array(self.moa.reshape((-1,1))))
        
        
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
        # Break the image file appropriately to read from the right folder
        img_file_split = img_file.split('_')
        week = img_file_split[0]
        sample = img_file_split[1]
        table = img_file_split[2]
        image_no = img_file_split[3]
        obj_no = img_file_split[4]

        # Read the image and transform it to a tensor 
        img = np.load(os.path.join(self.data_path, week, sample, table, image_no, f'{obj_no}.npy'))    
        img = torch.from_numpy(img).to(torch.float)
        img = img.permute(2,0,1)  # Place channel dimension in front of the others 
        img = self.transform(img)

        if self.return_labels:
            return dict(X=img, 
                        mol_one_hot=self.one_hot_drugs[idx],
                        moa_one_hot=self.one_hot_moa[idx],
                        smile_id=self.drugs2idx[self.mol_names[idx]],
                        moa_id=self.moa2idx[self.moa[idx]],
                        dose = self.dose[idx],
                        batch_number = sample,
                        table = table
                    )
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
        subset_moa_one_hot = []
        subset_smile_id = []
        subset_moa_id = []
        subset_dose = []
        
        for i in idx_sample:
            X, mol_one_hot, smile_id, moa_id, dose, _, _ = self.__getitem__(i).values()
            
            imgs.append(X.unsqueeze(0))  # Unsqueeze barch dimension
            subset_mol_one_hot.append(mol_one_hot)
            subset_smile_id.append(smile_id)
            subset_moa_id.append(moa_id)
            subset_dose.append(dose)

        imgs = torch.cat(imgs, dim=0)
        return dict(X = imgs,
                    mol_one_hot = subset_mol_one_hot,
                    moa_one_hot = subset_moa_one_hot, 
                    smile_id = subset_smile_id,
                    moa_id = subset_moa_id,
                    dose = self.dose[idx]) 

    
    def drug_or_moa_by_name(self, get_moa, name):
        """Load the images of specific drugs or moas by name

        Args:
            get_moa (bool): Whether the name refers to moa
            drug_name (str): The name of the drug of interest  
        """
        file_name_set = self.moa if get_moa else self.mol_names
        doses = []
        mols = []
        moas = []
        images = []
        idxs = [idx for idx in range(len(file_name_set)) if file_name_set[idx]==name]

        for idx in idxs:
            X, _, _, smile_id, moa_id, dose = self.__getitem__(idx).values()
            images.append(X.unsqueeze(0))
            doses.append(dose)
            mols.append(smile_id)
            moas.append(moa_id)

        return dict(X=torch.cat(images, dim=0), 
                drugs=mols,
                moas=moas,
                dose=doses) 
            
    def compute_class_imbalance_weights(self, class_vector):
        """Compute the sklearn class imbalance weight for a determined class vector 

        Args:
            class_vector (np.array): numpy array with the classes 
        """
        # Unique classes
        classes_unique = np.unique(class_vector)
        # Compute the weight vector to make the classes balanced 
        weight_vector = compute_class_weight('balanced', classes=classes_unique, y=class_vector)
        return weight_vector



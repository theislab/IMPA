import os 
import torch

from models.convVAE import *
from models.convAE import *
from models.sigmaVAE import *
from models.residconvVAE import *

from data.dataset import *
from utils import *
from torch.utils.data import Dataset
import torch
import itertools 
import numpy as np
import json
from tqdm import tqdm

#TODO: setup logger
#TODO: convert to checkpoint loading 
#TODO: implement early stopping 
#TODO: write description of the parameters 
#TODO: would be better to code the training within the respective classes to avoid too many conditionals 
 

class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(self.config_path) as f:
            self.params = json.load(f)
    
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            raise AttributeError("No such attribute: " + name)


class Trainer:
    """
    Class for training the model 
    """
    def __init__(self, config, train_mode = True, **kwargs):
        self.data_path = config.data_path
        self.num_epochs = config.num_epochs
        self.eval = config.eval
        self.eval_every = config.eval_every
        self.plot = config.plot
        self.normalize_imgs = config.normalize_imgs
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle
        self.n_workers_loader = config.n_workers_loader
        self.model_name = config.model_name
        self.model_config = config.model_config
        self.variational = config.variational 

        # Set device
        self.device = self.set_device() 
        print(f'Working on device: {self.device}')

        self.model =  self.load_model().to(self.device)
        self.checkpoint_path = config.checkpoint_path  # Where the model will be saved

        # Prepare the data
        print('Lodading the data...') 
        self.fold_datasets = self.read_folds()
        self.training_set, self.validation_set, self.test_set = self.create_torch_datasets()
        
        # Create data loaders 
        self.loader_train = torch.utils.data.DataLoader(self.training_set, batch_size=self.batch_size, shuffle=self.shuffle, 
                                                    num_workers=self.n_workers_loader, drop_last=False)
        self.loader_val = torch.utils.data.DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=self.shuffle, 
                                                    num_workers=self.n_workers_loader, drop_last=False)
        self.loader_test = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=self.shuffle, 
                                                    num_workers=self.n_workers_loader, drop_last=False)
        print('Successfully loaded the data')

        # Setup the optimizer 
        self.lr = config.lr
        self.wd = config.wd
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduling = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = config.step_size)


    def read_folds(self):
        """
        Extract the filenames of images in the train, test and validation sets from the 
        associated folder
        """
        # Get the file names and molecules of training, test and validation sets
        datasets = dict()
        for fold_name in ['train', 'val', 'test']:
            data_index_path = os.path.join(self.data_path, f'{fold_name}_data_index.npz')
            # Get the files with the sample splits and add them to the dictionary 
            fold_file, _, _ = get_files_and_mols_from_path(data_index_path=data_index_path)
            datasets[fold_name] = fold_file
        return datasets
    
    def create_torch_datasets(self):
        """
        Create dataset and data loader compatible with the pytorch training loop 
        """
        # Initialize data augmentation transform 
        transform = CustomTransform(normalize = self.normalize_imgs, test = False)
        # Create dataset objects for the three data folds 
        training_set = CellPaintingDataset(self.fold_datasets['train'], None, None, self.data_path, transform)
        transform.test = True
        validation_set = CellPaintingDataset(self.fold_datasets['val'], None, None, self.data_path, transform)
        test_set = CellPaintingDataset(self.fold_datasets['test'], None, None, self.data_path, transform)
        # Create the associated data loaders 
        return training_set, validation_set, test_set

    
    def train(self):
        """
        Full training 
        """
        print(f'Beginning training with epochs {self.num_epochs}')
        min_loss = np.inf
        for epoch in range(1, self.num_epochs+1):
            print(f'Running epoch {epoch}')
            self.model.train()
            losses = self.model.update_model(self.loader_train, epoch, self.optimizer, self.device) # Update run 

            # Scheduler step at the end of the epoch 
            self.lr_scheduling.step()

            # Evaluate
            if epoch % self.eval_every == 0:
                # Put the model in evaluate mode 
                self.model.eval()
                val_losses = self.model.evaluate(self.loader_val, self.validation_set, self.device)  
                val_loss = val_losses['loss']

                if self.plot:
                # Sample 
                    original  = next(iter(self.loader_val))[0].to(self.device).unsqueeze(0)  # Get the first element of the test batch 
                    with torch.no_grad():
                        reconstructed = self.model.generate(original)
                    self.plot_reconstruction(tensor_to_image(original), tensor_to_image(reconstructed))

                if val_loss < min_loss:
                    min_loss = val_loss
                    print(f'New best loss is {val_loss}')
                    # Save the model with lowest validation loss 
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    print(f'Save new checkpoint at {self.checkpoint_path}')


    def plot_reconstruction(self, original, reconstruction):
        """
        Plot the original and reconstructed channels one over the other 
        """
        fig = plt.figure(constrained_layout=True, figsize = (6,6))
        # create 3x1 subfigs
        subfigs = fig.subfigures(nrows=2, ncols=1)
        titles = ['ORIGINAL', 'RECONSTRUCTED']
        images = [original, reconstruction]        

        for row, subfig in enumerate(subfigs):
            subfig.suptitle(titles[row], fontsize = 20)

            # create 1x3 subplots per subfig
            axs = subfig.subplots(nrows=1, ncols=5)
            for col, ax in enumerate(axs):
                ax.imshow(images[row][:,:,col], cmap = 'Greys')
                ax.axis('off')
        plt.show()


    def set_device(self):
        """
        Set cpu or gpu device for data and computations 
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self):
        """
        Load the model from the dedicated library
        """
        models = {'ConvVAE':ConvVAE, 'ConvAE':ConvAE, 
        'ResidConvVAE':ResidConvVAE,'SigmaVAE': SigmaVAE}
        model = models[self.model_name]
        return model(**self.model_config)


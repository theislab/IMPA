import os 
import torch

from models.convVAE import *
from models.convAE import *
from models.sigmaVAE import *
from models.residconvVAE import *
from models.glow import *

from data.dataset import *
from utils import *
from plot_utils import Plotter 
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torch
import itertools 
import numpy as np
import json
from tqdm import tqdm
import time 

#TODO: convert to checkpoint loading 
#TODO: implement early stopping 
#TODO: write description of the parameters 
 

class Config:
    """
    Read a configuration file in .json format and wraps the hparams into a calss
    """
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
        self.normalize_imgs = config.normalize_imgs
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle
        self.n_workers_loader = config.n_workers_loader
        self.model_name = config.model_name
        self.model_config = config.model_config
        self.generate = config.generate
        self.temperature = config.temperature

        # Plot decisions
        self.experiment_name = config.experiment_name
        self.result_path = config.result_path
        self.img_plot = config.img_plot
        self.img_save = config.img_save
        print('Create output directories for the experiment')
        self.dest_dir = make_dirs(self.result_path, self.experiment_name)
        self.plotter = Plotter(self.dest_dir)

        # Setup logger
        self.writer = SummaryWriter(os.path.join(self.dest_dir, 'logs'))

        # Set device
        self.device = self.set_device() 
        print(f'Working on device: {self.device}')

        self.model =  self.load_model().to(self.device)
        self.model = nn.DataParallel(self.model)

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
        self.step_size = config.step_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.warmup_steps = config.warmup_steps

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
            # Losses from the epoch
            losses = self.model.module.update_model(self.loader_train, epoch, self.optimizer, self.device) # Update run 
            for key in losses:
                self.writer.add_scalar(tag=f'train/{key}', scalar_value=losses[key], global_step=epoch)

            # Evaluate
            if epoch % self.eval_every == 0:
                # Put the model in evaluate mode 
                self.model.eval()
                val_losses, metrics = self.model.module.evaluate(self.loader_val, self.validation_set, self.device)
                
                for key in val_losses:
                    self.writer.add_scalar(tag=f'val/{key}', scalar_value=val_losses[key], 
                                           global_step=epoch)
                for key in metrics:
                    self.writer.add_scalar(tag=f'val/{key}', scalar_value=metrics[key], global_step=epoch)

                val_loss = val_losses['loss']

                # Plot reconstruction
                original  = next(iter(self.loader_val))[0].to(self.device).unsqueeze(0)  # Get the first element of the test batch 
                with torch.no_grad():
                    reconstructed = self.model.module.generate(original)
                self.plotter.plot_reconstruction(tensor_to_image(original), 
                                                 tensor_to_image(reconstructed), epoch, self.img_save, self.img_plot)
                    
                # Plot generation of sampled images 
                if self.generate:
                    sampled_img = tensor_to_image(self.model.module.sample(1, self.temperature,
                                                                           self.device))
                    self.plotter.plot_channel_panel(sampled_img, epoch, self.img_save, self.img_plot)
                
                # Save the model if it is the best performing one 
                if val_loss < min_loss:
                    min_loss = val_loss
                    print(f'New best loss is {val_loss}')
                    # Save the model with lowest validation loss 
                    self.save_checkpoint()
                    print(f"Save new checkpoint at {os.path.join(self.dest_dir, 'checkpoint.pt')}")

            # Scheduler step at the end of the epoch 
            if epoch < self.warmup_steps:
                self.optimizer.param_groups[0]["lr"] = self.lr * min(1., self.warmup_steps/epoch)
            if epoch == self.warmup_steps-1:
                self.lr_scheduling = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                     step_size=self.step_size)
            elif epoch > self.warmup_steps-1:
                self.lr_scheduling.step()


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
        'ResidConvVAE':ResidConvVAE,'SigmaVAE': SigmaVAE,
        'Glow': Glow}
        model = models[self.model_name]
        return model(**self.model_config)
    
    def save_checkpoint(self):
        """
        Save the checkpoint after a training step 
        """
        checkpoint = 'checkpoint.pt'
        torch.save(self.model.state_dict(), os.path.join(self.dest_dir, checkpoint))
        


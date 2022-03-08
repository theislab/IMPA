import os
from random import sample 
import torch

from compert.vae.AE import *
from compert.vae.sigmaVAE import *
from compert.vae.VAE import *
from compert.flow.glow import *
from compert.vae_compert.VAEcompert import *

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
    def __init__(self, config, train_mode = True, **kwargs):
        """Class for training the model 

        Args:
            config (dict): The dictionary containing configuration parameters from .json file
        """
        self.experiment_name = config.experiment_name 

        self.image_path = config.image_path  # Path to the images 
        self.data_index_path = config.data_index_path  # Path to the image metadata
        self.embeddings_path = config.embedding_path  # The path to the molecular embedding csv
        self.result_path = config.result_path
        self.checkpoint_path = config.checkpoint_path

        # Resume the training 
        self.resume = config.resume 
        self.resume_checkpoint = config.resume_checkpoint
        self.resume_epoch = 1

        self.img_plot = config.img_plot
        self.img_save = config.img_save  # If the images has to be saved to the result folder

        self.num_epochs = config.num_epochs  # Number of epochs for training
        self.eval = config.eval  # Whether evaluation should occur
        self.eval_every = config.eval_every  # How often the images are evaluated  

        self.n_workers_loader = config.n_workers_loader # Number of workers for batch loading
        self.generate = config.generate  # Whetehr to perform a sampling + decoding experiment during evaluation
        self.model_name = config.model_name  # What model to run
        self.temperature = config.temperature  # Temperature to downscale the random samples from the prior
        self.augment_train = config.augment_train  # Whether augmentation should be carried out on the training set
        self.in_width = config.in_width
        self.in_height = config.in_height
        self.in_channels = config.in_channels
        self.adversarial = config.adversarial 
        self.binary_task = config.binary_task 
        self.append_layer_width = config.append_layer_width
        self.seed = config.seed 
    
        self.hparams = config.model_config  # Dictionary with model hyperparameters 

        # Set device
        self.device = self.set_device() 
        print(f'Working on device: {self.device}')

        self.model =  self.load_model().to(self.device)
        self.model = nn.DataParallel(self.model)
        
        # Prepare the data
        print('Lodading the data...') 
        self.training_set, self.validation_set, self.test_set, self.ood_set = self.create_torch_datasets()
        self.num_drugs = len(self.training_set.drug_encoder.categories_[0])
        
        # Create data loaders 
        self.loader_train = torch.utils.data.DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True, 
                                                    num_workers=self.n_workers_loader, drop_last=False)
        self.loader_val = torch.utils.data.DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=True, 
                                                    num_workers=self.n_workers_loader, drop_last=False)
        self.loader_test = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, 
                                                    num_workers=self.n_workers_loader, drop_last=False)
        self.loader_ood = torch.utils.data.DataLoader(self.ood_set, batch_size=self.batch_size, shuffle=True, 
                                                    num_workers=self.n_workers_loader, drop_last=False)
        print('Successfully loaded the data')


        # If training is resumed from a checkpoint
        if self.resume:
            self.resume_epoch, self.dest_dir = self.model.module.load_checkpoints(self.resume_checkpoint, self.optimizer, self.lr_scheduling)

    def create_torch_datasets(self):
        """
        Create dataset compatible with the pytorch training loop 
        """
        # Create dataset objects for the three data folds
        cellpainting_ds = CellPaintingDataset(self.image_path, self.data_index_path, self.embeddings_path, device=self.device, 
                                                return_labels=True, augment_train=self.augment_train)
        training_set, validation_set, test_set, ood_set = cellpainting_ds.fold_datasets.values()
        return training_set, validation_set, test_set, ood_set

    
    def train(self):
        """
        Full training 
        """
        # Create result folder
        if not self.resume and self.img_save:
            print('Create output directories for the experiment')
            self.dest_dir = make_dirs(self.result_path, self.experiment_name, self.img_save)
        
        # Setup plotter and writer 
        self.plotter = Plotter(self.dest_dir)

        # Setup logger
        self.writer = SummaryWriter(os.path.join(self.dest_dir, 'logs'))
        
        print(f'Beginning training with epochs {self.num_epochs}')
        min_loss = np.inf  # Will be updated at each step

        for epoch in range(self.resume_epoch, self.num_epochs+1):
            print(f'Running epoch {epoch}')
            self.model.train()
            # Losses from the epoch
            losses, metrics = self.model.module.update_model(self.loader_train, epoch) # Update run 
            for key in losses:
                self.writer.add_scalar(tag=f'train/{key}', scalar_value=losses[key], global_step=epoch)
            for key in metrics:
                self.writer.add_scalar(tag=f'train/{key}', scalar_value=metrics[key], global_step=epoch)

            # Evaluate
            if epoch % self.eval_every == 0:
                # Put the model in evaluate mode 
                self.model.eval()
                val_losses, metrics = self.model.module.evaluate(self.loader_val)
                
                for key in val_losses:
                    self.writer.add_scalar(tag=f'val/{key}', scalar_value=val_losses[key], 
                                           global_step=epoch)
                for key in metrics:
                    self.writer.add_scalar(tag=f'val/{key}', scalar_value=metrics[key], global_step=epoch)

                val_loss = val_losses['loss']

                # Plot reconstruction of a random image 
                original  = next(iter(self.loader_val))['X'][0].to(self.device).unsqueeze(0)  # Get the first element of the test batch 
                with torch.no_grad():
                    reconstructed = self.model.module.generate(original)
                self.plotter.plot_reconstruction(tensor_to_image(original), 
                                                 tensor_to_image(reconstructed), epoch, self.img_save, self.img_plot)
                del original
                del reconstructed
                    
                # Plot generation of sampled images 
                if self.generate:
                    sampled_img = tensor_to_image(self.model.module.sample(1, self.temperature))
                    self.plotter.plot_channel_panel(sampled_img, epoch, self.img_save, self.img_plot)
                    del sampled_img
                
                # Save the model if it is the best performing one 
                if val_loss < min_loss:
                    min_loss = val_loss
                    print(f'New best loss is {val_loss}')
                    # Save the model with lowest validation loss 
                    self.model.module.save_checkpoints(epoch, 
                                                        self.optimizer, 
                                                        self.lr_scheduling, 
                                                        metrics, 
                                                        losses, 
                                                        val_losses, 
                                                        self.checkpoint_path, 
                                                        self.dest_dir)

                    print(f"Save new checkpoint at {self.checkpoint_path}")

            # Scheduler step at the end of the epoch 
            if epoch <= self.warmup_steps:
                self.model.module.optimizer_autoencoder.param_groups[0]["lr"] = self.lr * min(1., self.model.module.warmup_steps/epoch)
            else:
                self.model.module.scheduler_autoencoder.step()
                self.model.module.scheduler_adversary.step()


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
        # Dictionary of models 
        models = {'compertVAE': compertVAE}
        model = models[self.model_name]
        return model(adversarial = self.adversarial,
                    in_width = self.in_width,
                    in_height = self.in_height,
                    in_channels = self.in_channels,
                    device = self.device,
                    num_drugs = self.num_drugs,
                    seed = self.seed,
                    patience = self.patience,
                    hparams = self.hparams,
                    binary_task = self.binary_task,
                    append_layer_width = self.append_layer_width
                    )

# Auziliary functions 

def gaussian_nll(mu, log_sigma, x):
    """
    Implement Gaussian nll loss
    """
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)
    return result_tensor

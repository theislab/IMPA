import os
import torch

# Available autoencoder model attached to CPA 
from compert.model.sigma_VAE import SigmaVAE
from compert.model.sigma_AE import SigmaAE
from data.dataset import CellPaintingDataset
from compert.utils import *
from compert.plot_utils import Plotter 
from compert.evaluate import *

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import json


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
    def __init__(self, config):
        """Class for training the model 

        Args:
            config (dict): The dictionary containing configuration parameters from .json file
        """
        self.experiment_name = config.experiment_name  # Will be used to name the path 

        self.image_path = config.image_path  # Path to the images 
        self.data_index_path = config.data_index_path  # Path to the image metadata
        self.embeddings_path = config.embedding_path  # The path to the molecular embedding csv
        self.use_embeddings = config.use_embeddings  # Whether to use embeddings or not

        # Path to save the results
        self.result_path = config.result_path

        # Resume the training 
        self.resume = config.resume 
        self.resume_checkpoint = config.resume_checkpoint
        self.resume_epoch = 1

        self.img_plot = config.img_plot
        self.save_results = config.save_results  # If the images has to be saved to the result folder

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

        self.patience = config.patience
        self.seed = config.seed 
    
        self.hparams = config.hparams  # Dictionary with model hyperparameters 

        # Set device
        self.device = self.set_device() 
        print(f'Working on device: {self.device}')

        
        # Prepare the data
        print('Lodading the data...') 
        self.drug_embeddings, self.num_drugs, self.n_seen_drugs, self.training_set, self.validation_set, self.test_set, self.ood_set = self.create_torch_datasets()

        
        # Initialize model 
        self.model =  self.load_model().to(self.device)
        self.model = torch.nn.DataParallel(self.model)
        
        # Create data loaders 
        self.loader_train = torch.utils.data.DataLoader(self.training_set, batch_size=self.hparams["batch_size"], shuffle=True, 
                                                    num_workers=self.n_workers_loader, drop_last=False)
        self.loader_val = torch.utils.data.DataLoader(self.validation_set, batch_size=1, shuffle=True, 
                                                    num_workers=self.n_workers_loader, drop_last=False)
        self.loader_test = torch.utils.data.DataLoader(self.test_set, batch_size=1, shuffle=True, 
                                                    num_workers=self.n_workers_loader, drop_last=False)
        self.loader_ood = torch.utils.data.DataLoader(self.ood_set, batch_size=1, shuffle=True, 
                                                    num_workers=self.n_workers_loader, drop_last=False)
        print('Successfully loaded the data')


        # Create result folder
        if self.save_results:
            if not self.resume:
                print('Create output directories for the experiment')
                self.dest_dir = make_dirs(self.result_path, self.experiment_name)
                # Setup logger
                self.writer = SummaryWriter(os.path.join(self.dest_dir, 'logs'))
            # If training is resumed from a checkpoint
            if self.resume:
                self.resume_epoch, self.dest_dir = self.model.module.load_checkpoints(self.resume_checkpoint, self.adversarial)


    def create_torch_datasets(self):
        """
        Create dataset compatible with the pytorch training loop 
        """
        # Create dataset objects for the three data folds
        cellpainting_ds = CellPaintingDataset(self.image_path, self.data_index_path, self.embeddings_path, device=self.device, 
                                                return_labels=True, use_pretrained=self.use_embeddings, augment_train=self.augment_train)
        if self.use_embeddings:
            drug_embeddings = cellpainting_ds.drug_embeddings
        else:
            drug_embeddings = None
        num_drugs = cellpainting_ds.num_drugs
        n_seen_drugs = cellpainting_ds.n_seen_drugs
        training_set, validation_set, test_set, ood_set = cellpainting_ds.fold_datasets.values()
        return drug_embeddings, num_drugs, n_seen_drugs, training_set, validation_set, test_set, ood_set

    
    def train(self):
        """
        Full training 
        """
        if self.img_plot:
            # Setup plotter and writer 
            self.plotter = Plotter(self.dest_dir)
        
        # The variable `end` determines whether we are at the end of the training loop and therefore if disentanglement and clustering stats
        # are to be evaluated
        end = False

        print(f'Beginning training with epochs {self.num_epochs}')

        for epoch in range(self.resume_epoch, self.num_epochs+1):
            print(f'Running epoch {epoch}')
            self.model.train()
            self.model.module.metrics.mode = 'train'
            # Losses from the epoch
            losses, metrics = self.model.module.update_model(self.loader_train, epoch)  # Update run 
            
            # Save results to tensorboard and to model's history  
            if self.save_results:
                self.write_results(losses, metrics, self.writer, epoch, 'train')
            self.model.module.save_history(epoch, losses, metrics, 'train')
            
            # Evaluate
            if epoch % self.eval_every == 0 and self.eval:
                # Put the model in evaluate mode 
                self.model.eval()
                self.model.module.metrics.mode = 'val'
                val_losses, metrics = training_evaluation(self.model.module, self.loader_val, self.adversarial,
                                                        self.model.module.metrics, self.binary_task, self.device, end, 
                                                        variational=self.model.module.variational)

                if self.save_results:
                    self.write_results(val_losses, metrics, self.writer, epoch, 'val')
                self.model.module.save_history(epoch, val_losses, metrics, 'val')

                # Plot reconstruction of a random image 
                if self.img_plot:
                    with torch.no_grad():
                        original, reconstructed = self.model.module.generate(self.loader_val)
                    self.plotter.plot_reconstruction(tensor_to_image(original), 
                                                    tensor_to_image(reconstructed), epoch, self.save_results, self.img_plot)
                    del original
                    del reconstructed
                    
                    # Plot generation of sampled images 
                    if self.generate:
                        sampled_img = tensor_to_image(self.model.module.sample(1, self.temperature))
                        self.plotter.plot_channel_panel(sampled_img, epoch, self.save_results, self.img_plot)
                        del sampled_img

                losses = {'a':242}
                # Decide on early stopping based on the bit/dim of the image 
                score = metrics['bpd']
                cond, early_stopping = self.model.module.early_stopping(score) 
                
                # Save the model if it is the best performing one 
                if cond and self.save_results:
                    print(f'New best score is {self.model.module.best_score}')
                    # Save the model with lowest validation loss 
                    self.model.module.save_checkpoints(epoch, 
                                                        metrics, 
                                                        losses, 
                                                        val_losses, 
                                                        self.dest_dir, self.adversarial)

                    print(f"Save new checkpoint at {os.path.join(self.dest_dir, 'checkpoint')}")

            # If we overcome the patience, we break the loop
            if early_stopping:
                break 

            # Scheduler step at the end of the epoch 
            if epoch <= self.hparams["warmup_steps"]:
                self.model.module.optimizer_autoencoder.param_groups[0]["lr"] = self.hparams["autoencoder_lr"] * min(1., self.model.module.warmup_steps/epoch)
            else:
                self.model.module.scheduler_autoencoder.step()
            
            if self.adversarial:
                self.model.module.scheduler_adversaries.step()
    
        # Perform last evaluation on VALIDATION SET
        end = True
        val_losses, metrics = training_evaluation(self.model.module, self.loader_val, self.adversarial,
                                                self.model.module.metrics, self.binary_task, self.device, end, 
                                                variational=self.model.module.variational)
        if self.save_results:
            self.write_results(val_losses, metrics, self.writer, epoch, ' val')
        self.model.module.save_history('final', val_losses, metrics, 'val')

        # Perform last evaluation on OOD SET
        ood_losses, metrics = training_evaluation(self.model.module, self.loader_ood, adversarial=False,
                                                metrics=self.model.module.metrics, binary_task=self.binary_task, device=self.device, end=end, 
                                                variational=self.model.module.variational)
        if self.save_results:
            self.write_results(ood_losses, metrics, self.writer, epoch,' ood')
        self.model.module.save_history('final', losses, metrics, 'ood')
        


        

    def write_results(self, losses, metrics, writer, epoch, fold='train'):
        """Write results to tensorboard

        Args:
            losses (dict): _description_
            metrics (dict): _description_
            writer (torch.utils.tensorboard.SummaryWriter): summary statistics writer 
        """
        for key in losses:
            writer.add_scalar(tag=f'{fold}/{key}', scalar_value=losses[key], 
                                    global_step=epoch)
        for key in metrics:
            writer.add_scalar(tag=f'{fold}/{key}', scalar_value=metrics[key], global_step=epoch)

        
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
        models = {'VAE': SigmaVAE, 'AE': SigmaAE}
        model = models[self.model_name]
        return model(adversarial = self.adversarial,
                    in_width = self.in_width,
                    in_height = self.in_height,
                    in_channels = self.in_channels,
                    device = self.device,
                    num_drugs = self.num_drugs,
                    n_seen_drugs = self.n_seen_drugs,
                    seed = self.seed,
                    patience = self.patience,
                    hparams = self.hparams,
                    binary_task = self.binary_task,
                    append_layer_width = self.append_layer_width,
                    drug_embeddings = self.drug_embeddings)


# Auxiliary functions 

def gaussian_nll(mu, log_sigma, x):
    """
    Implement Gaussian nll loss
    """
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)
    return result_tensor

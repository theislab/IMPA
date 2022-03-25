import torch
from torch import nn

import sys 
sys.path.insert(0, '..')

from metrics.metrics import *
import os

"""
Basic model class from which all others inherit to load and save the checkpoints
"""

class TemplateModel(nn.Module):
    def __init__(self):
        super(TemplateModel, self).__init__() 
        pass
    
    def save_checkpoints(self, 
                        epoch, 
                        metrics, 
                        train_losses, 
                        val_losses, 
                        dest_dir):
        """
        Save the checkpoints to a checkpoint dict
        """
        checkpoint = dict()
        # Save to state dict
        checkpoint['epoch'] = epoch
        checkpoint['model_state_dict'] = self.state_dict()  # Parameters of the model
        checkpoint['optimizer_autoencoder'] = self.optimizer_autoencoder.state_dict()  # Optimizer autoencoder
        checkpoint['scheduler_autoencoder'] = self.scheduler_autoencoder.state_dict()  # Scheduler
        checkpoint['metrics'] = metrics  # Validation metrics 
        checkpoint['train_losses'] = train_losses  # Training losses 
        checkpoint['val_losses'] = val_losses  # Validation losses
        checkpoint['dest_dir'] = dest_dir
        checkpoint['history'] = self.history
        if self.adversarial:
            checkpoint['optimizer_adversaries'] = self.optimizer_adversaries.state_dict()  # Optimizer autoencoder
            checkpoint['scheduler_autoencoder'] = self.scheduler_autoencoder.state_dict()  # Scheduler
        torch.save(checkpoint, os.path.join(dest_dir, 'checkpoints','checkpoint.pt'))    


    def load_checkpoints(self, checkpoint_path):
        """
        Load the checkpoints from a checkpoint path 
        """
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_autoencoder.load_state_dict(checkpoint['optimizer_autoencoder'])
        self.scheduler_autoencoder.load_state_dict(checkpoint['scheduler_autoencoder'])
        metrics = checkpoint['metrics']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        dest_dir = checkpoint['dest_dir']
        self.history = checkpoint['history']
        if 'optimizer_adversaries' in checkpoint:
            self.optimizer_adversaries.load_state_dict(checkpoint['optimizer_adversaries'])    # Optimizer autoencoder
            self.scheduler_adversaries.load_state_dict(checkpoint['scheduler_autoencoder'])    # Scheduler

        print(f'Loaded model at epoch {epoch}')
        for loss in train_losses:
            print(f'Train {loss}: {train_losses[loss]}')
        for loss in val_losses:
            print(f'Validation {loss}: {val_losses[loss]}')
        for metric in metrics:
            print(f'{metric}: {metrics[metric]}')
        
        return epoch, dest_dir
    
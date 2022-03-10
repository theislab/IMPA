import torch
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm
import numpy as np
from compert.training_utils import *
from metrics import *
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
                        optimizer, 
                        scheduler, 
                        metrics, 
                        train_losses, 
                        val_losses, 
                        checkpoint_path, 
                        dest_dir):
        """
        Save the checkpoints to a checkpoint dict
        """
        checkpoint = dict()
        
        # Save to state dict
        checkpoint['epoch'] = epoch
        checkpoint['model_state_dict'] = self.state_dict()  # Parameters of the model
        checkpoint['optimizer'] = optimizer.state_dict()  # Optimizer
        checkpoint['scheduler'] = scheduler.state_dict()  # Scheduler
        checkpoint['metrics'] = metrics  # Validation metrics 
        checkpoint['train_losses'] = train_losses  # Training losses 
        checkpoint['val_losses'] = val_losses  # Validation losses
        checkpoint['dest_dir'] = dest_dir
        torch.save(checkpoint, os.path.join(checkpoint_path))        

    def load_checkpoints(self, checkpoint_path, optimizer, scheduler):
        """
        Load the checkpoints from a checkpoint path 
        """
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        metrics = checkpoint['metrics']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        dest_dir = checkpoint['dest_dir']

        print(f'Loaded model at epoch {epoch}')
        for loss in train_losses:
            print(f'Train {loss}: {train_losses[loss]}')
        for loss in val_losses:
            print(f'Validation {loss}: {val_losses[loss]}')
        for metric in metrics:
            print(f'{metric}: {metrics[metric]}')
        
        return epoch, dest_dir
    




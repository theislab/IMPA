import sys 
sys.path.insert(0, '..')

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

class TrainingMetrics:
    def __init__(self, height, width, channels, batch_size, device='cuda'):
        self.height = height
        self.width = width
        self.channels = channels

        self.metrics = dict(rmse=0) 
        self.iterations = dict(rmse=0)
        self.device = device
        self.batch_size = batch_size 

    def update_rmse(self, X, X_hat):
        """
        Update RMSE with the result from the batch 
        """
        avg = self.compute_batch_rmse(X, X_hat)
        # Add the sum of rmses per batch
        self.metrics['rmse'] += avg.item()
        self.iterations['rmse'] += 1
    
    def compute_batch_rmse(self, X, X_hat):
        """
        Compute RMSE with the result from the batch 
        """
        # Pick the squared difference between two tensor objects
        diff = (X - X_hat)**2
        # Take the mean across spatial and channel dimension 
        return torch.mean(torch.sqrt(diff.mean(dim=(1, 2, 3))))
    
    def update_bpd(self, loss):
        """
        Update bpd with the result from the batch 
        """
        self.metrics['bpd'] = (-loss)/(self.height*self.width*self.channels)/np.log(2.)
        self.metrics['bpd'] = self.metrics['bpd'].item()
    
    def compute_classification_report(self, y, y_hat, label=''):
        """
        Compute the precision, recall and F1 of two ground truth vectors
        """
        prec, rec, F1, _ = precision_recall_fscore_support(y, y_hat, average='macro', zero_division=0)
        self.metrics[f'precision{label}'] = prec 
        self.metrics[f'recall_label{label}'] = rec
        self.metrics[f'F1{label}'] = F1
        
    def get_metrics(self):
        """
        Return the metrics dictionary 
        """
        return self.metrics

    def reset(self):
        """
        Reset metrics to 0        
        """
        self.metrics = dict(rmse=0)  
        self.iterations = dict(rmse=0)  

    def print_metrics(self):
        """
        Print each element of the metrics vector 
        """
        for metric in self.metrics:
            print(f'{metric} = {self.metrics[metric]}')

    def average_metrics(self):
        """
        Compute the batch average for each metric         
        """
        for key in self.metrics:
            self.metrics[key] = self.metrics[key]/self.iterations[key]


class TrainingLosses:
    def __init__(self):
        self.loss_dict = None
        self.iteration_dict = None

    def initialize_losses(self, losses):
        """
        Initialize the loss vector attribute 
        """
        self.loss_dict = {}
        self.iteration_dict = {}
        for key in losses:
            self.loss_dict[key] = 0
            self.iteration_dict[key] = 0
        
    def update_losses(self, losses):
        """
        Update the vector of losses 
        """
        # Initialize losses if they don't exist yet 
        if self.loss_dict == None:
            self.initialize_losses(losses)
        # Record the losses
        for loss in losses:
            # If loss is not in the dict, update it and its iteration dict 
            if loss not in self.loss_dict:
                try: 
                    self.loss_dict[loss] = losses[loss].item()
                except:
                    self.loss_dict[loss] = losses[loss]
                self.iteration_dict[loss] = 1
                
            else:
                try:
                    self.loss_dict[loss] += losses[loss].item()
                except:
                    self.loss_dict[loss] += losses[loss]
                self.iteration_dict[loss] += 1

    def reset(self):
        self.loss_dict = None 
        self.iteration_dict = None 

    def print_losses(self):
        for loss in self.loss_dict:
            print(f'Average {loss} = {self.loss_dict[loss]}')
    
    def average_losses(self):
        """
        Convert the losses to batch averages            
        """
        for key in self.loss_dict:
            self.loss_dict[key] = self.loss_dict[key]/self.iteration_dict[key]

import sys 
sys.path.insert(0, '..')

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

class TrainingMetrics:
    def __init__(self, height, width, channels, latent_dimension, device='cuda'):
        self.height = height
        self.width = width
        self.channels = channels
        self.latent_dimension = latent_dimension

        self.metrics = dict(rmse=0) 
        self.device = device


    def update_rmse(self, X, X_hat):
        """
        Update RMSE with the result from the batch 
        """
        avg = self.compute_batch_rmse(X, X_hat)
        # Add the sum of rmses per batch
        self.metrics['rmse'] += avg.item()
    

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
    

    def compute_classification_report(self, y, y_hat):
        prec, rec, F1, _ = precision_recall_fscore_support(y, y_hat, average='macro', zero_division=0)
        self.metrics['precision'] = prec 
        self.metrics['recall'] = rec
        self.metrics['F1'] = F1
        

    def get_metrics(self):
        return self.metrics


    def reset(self):
        self.metrics = dict(rmse=0)    

          
    def print_metrics(self):
        for metric in self.metrics:
            print(f'{metric} = {self.metrics[metric]}')


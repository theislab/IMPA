import torch
import numpy as np
import torch.nn.functional as F

class TrainingMetrics:
    def __init__(self, height, width, channels, latent_dimension, flow=False, device='cuda'):
        self.height = height
        self.width = width
        self.channels = channels
        self.latent_dimension = latent_dimension
        self.flow = flow

        if not self.flow:
            self.mse_metric = torch.nn.MSELoss()
            self.metrics = dict(rmse=0, bpd=0, batches = 0) 
        else:
            self.metrics = dict( batches = 0) 
        self.device = device

    def update_rmse(self, X, X_hat):
        """
        Update RMSE with the result from the batch 
        """
        self.metrics['rmse'] += torch.sqrt(self.mse_metric(X, X_hat)).detach()
    
    def update_bpd(self, loss):
        """
        Update bpd with the result from the batch 
        """
        self.metrics['bpd'] = (-loss)/(self.height*self.width*self.channels)/np.log(2.)

    def get_metrics(self):
        return self.metrics

    def reset(self):
        for metric in self.metrics:
            self.metrics[metric] = 0        
          
    def print_metrics(self, flow = False):
        if not self.flow:
            print(f'Rmse = {self.metrics["rmse"]}')
            print(f'LLH in bit/dim = {self.metrics["bpd"]}')
        




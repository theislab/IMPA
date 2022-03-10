import torch
import numpy as np
import sklearn 
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

class TrainingMetrics:
    def __init__(self, height, width, channels, latent_dimension, mode='val', device='cuda'):
        self.height = height
        self.width = width
        self.channels = channels
        self.latent_dimension = latent_dimension
        self.mode = mode

        self.mse_metric = torch.nn.MSELoss()
        self.metrics = dict(rmse=0, bpd=0, batches = 0) 

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
    
    def compute_classification_report(self, y, y_hat):
        prec, rec, F1, _ = precision_recall_fscore_support(y, y_hat, average='macro', zero_division=0)
        self.metrics['precision'] = prec 
        self.metrics['recall'] = rec
        self.metrics['F1'] = F1
        

    def get_metrics(self):
        return self.metrics


    def reset(self):
        for metric in self.metrics:
            self.metrics[metric] = 0        

          
    def print_metrics(self, flow = False):
        for metric in self.metrics:
            print(f'{metric} = {self.metrics[metric]}')





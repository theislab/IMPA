import torch

class Metrics:
    def __init__(self):
        self.mse_metric = torch.nn.MSELoss()
        self.metrics = dict(rmse=0) 

    def metrics_update(self, X, X_hat):
        """
        Update RMSE with the result from the batch 
        """
        self.metrics['rmse'] += torch.sqrt(self.mse_metric(X, X_hat))
    
    def get_metrics(self):
        return self.rmse

    def reset(self):
        self.metrics['rmse'] = 0          
        
    def print_metrics(self):
        for key in self.metrics:
            print(f'The validation {key} is {self.metrics[key]}')
        




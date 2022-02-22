import torch
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm
import numpy as np
from training_utils import *
from metrics import *
from models.vae.VAEmodel import *


class VAE(BasicVAE):
    def __init__(self,
                in_channels: int = 5,
                latent_dim: int = 512,
                hidden_dims: list = None,
                n_residual_blocks: int = 6, 
                in_width: int = 64,
                in_height: int = 64,
                resnet = True,
                device = 'cuda',
                **kwargs) -> None:

        super(VAE, self).__init__()   

        # Log-scale for the likelihood
        self.metrics = Metrics()
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))


    def reconstruction_loss(self, X_hat, X):
        """ 
        Computes the likelihood of the data given the latent variable,
        in this case using a Gaussian distribution with mean predicted by the neural network and variance = 1 
        """
        # Gaussian log lik
        rec = gaussian_nll(X_hat, self.log_scale, X).sum((1,2,3)).mean()
        return rec

    
    def loss_function(self, X, X_hat, mu, log_sigma, **kwargs) -> dict:
        """
        Computes the VAE loss 
        X: The input data 
        X_hat: The reconstucted input 
        mu: the mean encodings
        log_sigma: the log variance encodings 
        """
        # Reconstruction loss between the input and the prediction
        rec = self.reconstruction_loss(X_hat, X)

        # KL divergence 
        kl = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim=1), dim=0)
        
        # Total loss 
        loss = rec + kl
        # Zero out the gradient of the losses 
        return {'loss': loss, 'Reconstruction_Loss': rec.detach(), 'KLD': kl.detach()}
        

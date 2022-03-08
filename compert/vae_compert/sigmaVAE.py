import torch
import torch.utils.data
from torch import nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.models as models
from training_utils import *
from models.vae.VAEmodel import *


class SigmaVAE(BasicVAE):
    def __init__(self,
                in_channels,
                latent_dim,
                hidden_dims,
                n_residual_blocks, 
                in_width,
                in_height,
                device='cuda',
                **kwargs) -> None:
     
        super(SigmaVAE, self).__init__(in_channels, latent_dim, hidden_dims, n_residual_blocks, in_width, in_height, device)

    def reconstruction_loss(self, X_hat, X):
        """ Computes the likelihood of the data given the latent variable,
        in this case using a Gaussian distribution with mean predicted by the neural network and variance = 1 """
        # log rmse
        log_scale = ((X - X_hat) ** 2).mean([0,1,2,3], keepdim=True).sqrt().log()  # Keep the 3 dimensions 
        self.log_scale = torch.nn.Parameter(log_scale)
        # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
        # ensures stable training.
        log_scale = softclip(log_scale, -6)
        # Gaussian log lik
        rec = gaussian_nll(X_hat, log_scale, X).sum((1,2,3)).mean()  # Single value (not averaged across batch element)
        return rec

    def loss_function(self, X_hat, X, mu, log_sigma):
        """
        Aggregate the reconstruction and kl losses 
        """
        rec = self.reconstruction_loss(X_hat, X)
        kl = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim = 1), dim = 0)
        loss = rec + kl
        return {'loss': loss, 'Reconstruction_Loss': rec.detach(), 'KLD': kl.detach()}


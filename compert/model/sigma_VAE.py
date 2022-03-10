import torch
import torch.utils.data
from torch import nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.models as models
from compert.training_utils import *
from compert.model.CPA import *


class SigmaVAE(CPA):
    def __init__(self,
                adversarial: bool,
                in_width: int,
                in_height: int,
                in_channels: int,
                device: str,
                num_drugs: int,
                n_seen_drugs: int,
                seed: int = 0,
                patience: int = 5,
                hparams="",
                binary_task=False,
                append_layer_width=None,
                drug_embeddings = None) -> None:
     
        super(SigmaVAE, self).__init__(adversarial,
                                        in_width,
                                        in_height,
                                        in_channels,
                                        device,
                                        num_drugs,
                                        n_seen_drugs,   
                                        seed,
                                        patience,
                                        hparams,
                                        binary_task,
                                        append_layer_width,
                                        drug_embeddings)

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

    def vae_loss(self, X, X_hat, mu, log_sigma):
        """
        Aggregate the reconstruction and kl losses 
        """
        rec = self.reconstruction_loss(X_hat, X)
        kl = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim = 1), dim = 0)
        loss = rec + kl
        return {'loss': loss, 'Reconstruction_Loss': rec.detach(), 'KLD': kl.detach()}


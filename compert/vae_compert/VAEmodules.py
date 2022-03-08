import torch
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm
import numpy as np



def gaussian_nll(mu, log_sigma, x):
    """
    Compute the Gaussian negative log-likelihood loss
    
    mu: mean
    log_sigma: log standard deviation
    x: observation
    """
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

# Softclip
def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


"""
Convolutional layer with residual connection 
"""

# Convolutional layer with residual connection 
class ResidualLayer(nn.Module):
    """
    Simple residual block 
    """
    def __init__(self, in_channels, out_channel):
        super(ResidualLayer, self).__init__()
        # Residual unit 
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size = 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size = 1)
            )
        self.activation_out = nn.LeakyReLU()

    def forward(self, X):
        out = self.resblock(X)
        out += X  # Residual connection 
        out = self.activation_out(out)
        return out

"""
Encoder and Decoder classes 
"""

class Encoder(nn.Module):
    def __init__(self,
                in_channels: int = 5,
                latent_dim: int = 512,
                hidden_dim: int = 64,
                depth: int = 3,
                n_residual_blocks: int = 6, 
                in_width: int = 64,
                in_height: int = 64,
                **kwargs) -> None:
        super(Encoder, self).__init__() 
    
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim  # First number of feature maps
        self.depth = depth 
        self.n_residual_blocks = n_residual_blocks
        self.in_width, self.in_height = in_width, in_height
        self.kernel_size = 3

        # List containing the modules 
        self.modules = []

        # First layer - no downsizing 
        self.modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=self.hidden_dim,
                                kernel_size=(self.kernel_size, self.kernel_size), 
                                stride=1, padding=1),
                    nn.BatchNorm2d(self.hidden_dims[0]),
                    nn.ReLU())
            )        
        self.kernel_size += 1

        # Build downsizing convolutions 
        in_channels = self.hidden_dims
        for _ in range(1, self.depth):
            self.modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=in_channels*2,
                                kernel_size=self.kernel_size, 
                                stride=2, padding=(self.kernel_size-1)//2),
                    nn.ReLU())
            )
            self.kernel_size += 1
            in_channels = in_channels*2
        
        # Add residual blocks 
        for _ in range(self.n_residual_blocks):
            self.modules.append(ResidualLayer(in_channels, in_channels))

        self.encoder = nn.Sequential(*self.modules)

        # Add bottleneck
        downsampling_factor_width = self.in_width//2**(self.depth-1)
        downsampling_factor_height = self.in_height//2**(self.depth-1)
        self.flattened_dim = in_channels*downsampling_factor_width*downsampling_factor_height
        self.flatten = nn.Flatten()
        
        self.fc_mu = nn.Linear(self.flattened_dim, self.latent_dim)  # Mean encoding
        self.fc_var = nn.Linear(self.flattened_dim, self.latent_dim)  # Log-var encodings 

    def forward(self, X):
        X = self.encoder(X)  # Encode the image 
        X = self.flatten(X)  

        # Derive the encodings for the mean and the log variance
        mu = self.fc_mu(X)  
        log_sigma = self.fc_var(X)
        return [mu, log_sigma]


class Decoder(nn.Module):
    def __init__(self,
                out_channels: int = 5,
                latent_dim: int = 512,
                hidden_dim: int = 64,
                depth: int = 3,
                n_residual_blocks: int = 6, 
                out_width: int = 64,
                out_height: int = 64,
                **kwargs) -> None:

        super(Decoder, self).__init__() 
        
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim*(2**(self.depth-1))  # The first number of feature vectors 
        self.depth = depth 
        self.n_residual_blocks = n_residual_blocks
        self.out_width, self.out_height = out_width, out_height
        self.kernel_size = 6

        # Build convolutional dimensions
        self.modules = []

        # Layer to upscale latent sample 
        self.upsampling_factor_width = self.out_width//2**(self.depth-1)
        self.upsampling_factor_height = self.out_height//2**(self.depth-1)
        self.flattened_dim = self.hidden_dim*self.upsampling_factor_width*self.upsampling_factor_height
        self.upsample_fc = nn.Linear(self.latent_dim, self.flattened_dim)

        # Append the residual blocks
        for _ in range(self.n_residual_blocks):
            self.modules.append(ResidualLayer(self.hidden_dim, self.hidden_dim))

        # First deconvolution with BN    
        self.modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim//2,
                            kernel_size=self.kernel_size, stride=2, padding=2),
                nn.BatchNorm2d(self.hidden_dim*2),
                nn.ReLU()),
                )    

        in_channels = self.hidden_dim//2

        # Extra deconvolutions without batch normalization
        for _ in range(1, self.depth-1):
            self.modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, in_channels//2,
                                kernel_size=self.kernel_size, stride=2, padding=2),
                    nn.ReLU())
                    )  

            self.kernel_size -= 1
            in_channels = in_channels//2
        
        self.modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(in_channels, self.out_channels,
                            kernel_size=self.kernel_size, stride=1, padding=2),
                nn.Sigmoid())
                )   

        self.decoder = nn.Sequential(*self.modules)
    
    def forward(self, z):
        X = self.upsample_fc(z)
        # Reshape to height x width
        X = X.view(-1, self.hidden_dim, self.upsampling_factor_width, self.upsampling_factor_height)
        X = self.decoder(X)
        return X 

import torch
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm
import numpy as np
from training_utils import *
from models.model import TemplateModel
from metrics.metrics import *

# Gaussian negative log-likelihood
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
Layers 
"""

# Flatten and unflatten layers
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, n_channels):
        super(UnFlatten, self).__init__()
        self.n_channels = n_channels
    
    def forward(self, input):
        size = int((input.size(1) // self.n_channels) ** 0.5)
        return input.view(input.size(0), self.n_channels, size, size)


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
                hidden_dims: list = None,
                n_residual_blocks: int = 6, 
                in_width: int = 64,
                in_height: int = 64,
                **kwargs) -> None:
        super(Encoder, self).__init__() 
    
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.n_residual_blocks = n_residual_blocks
        self.in_width, self.in_height = in_width, in_height
        self.kernel_size = 3

        # If not pre-specified, convolutional layers have 32, 64 and 128 feature maps
        self.modules = []
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 256]
        
        # First layer - no downsizing 
        self.modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=self.hidden_dims[0],
                                kernel_size=(self.kernel_size, self.kernel_size), 
                                stride=1, padding=1),
                    nn.BatchNorm2d(self.hidden_dims[0]),
                    nn.ReLU())
            )        
        self.kernel_size += 1
        in_channels = self.hidden_dims[0]
        
        # Downsizing convolutional blocks 
        for h_dim in self.hidden_dims[1:]:
            self.modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                                kernel_size=self.kernel_size, 
                                stride=2, padding=(self.kernel_size-1)//2),
                    nn.ReLU())
            )
            self.kernel_size += 1
            in_channels = h_dim
        
        # Add residual blocks 
        for _ in range(self.n_residual_blocks):
            self.modules.append(ResidualLayer(self.hidden_dims[-1], self.hidden_dims[-1]))

        self.encoder = nn.Sequential(*self.modules)

        # Add bottleneck
        downsampling_factor_width = self.in_width//2**(len(self.hidden_dims)-1)
        downsampling_factor_height = self.in_height//2**(len(self.hidden_dims)-1)

        self.flattened_dim = self.hidden_dims[-1]*downsampling_factor_width*downsampling_factor_height
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
                hidden_dims: list = None,
                n_residual_blocks: int = 6, 
                out_width: int = 64,
                out_height: int = 64,
                **kwargs) -> None:

        super(Decoder, self).__init__() 
        
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.n_residual_blocks = n_residual_blocks
        self.out_width, self.out_height = out_width, out_height
        self.kernel_size = 6

        # Build convolutional dimensions
        self.modules = []
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        self.hidden_dims = hidden_dims

        # Layer to upscale latent sample 
        self.upsampling_factor_width = self.out_width//2**(len(self.hidden_dims)-1)
        self.upsampling_factor_height = self.out_height//2**(len(self.hidden_dims)-1)
        self.flattened_dim = self.hidden_dims[0]*self.upsampling_factor_width*self.upsampling_factor_height
        self.upsample_fc = nn.Linear(self.latent_dim, self.flattened_dim)

        # Append the residual blocks
        for _ in range(self.n_residual_blocks):
            self.modules.append(ResidualLayer(self.hidden_dims[0], self.hidden_dims[0]))

        # First deconvolution with BN    
        self.modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dims[0], self.hidden_dims[1],
                            kernel_size=self.kernel_size, stride=2, padding=2),
                nn.BatchNorm2d(self.hidden_dims[1]),
                nn.ReLU())
                )    

        in_channels = self.hidden_dims[1]

        # Extra deconvolutions without batch normalization
        for h_dim in self.hidden_dims[2:]:
            self.modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, h_dim,
                                kernel_size=self.kernel_size, stride=2, padding=2),
                    nn.ReLU())
                    )  
            self.kernel_size -= 1
            in_channels = h_dim
        
        self.modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dims[-1], self.out_channels,
                            kernel_size=self.kernel_size, stride=1, padding=2),
                nn.Sigmoid())
                )   

        self.decoder = nn.Sequential(*self.modules)
    
    def forward(self, z):
        X = self.upsample_fc(z)
        # Reshape to height x width
        X = X.view(-1, self.hidden_dims[0], self.upsampling_factor_width, self.upsampling_factor_height)
        X = self.decoder(X)
        return X 


"""
The default AE/VAE class 
"""

class BasicVAE(TemplateModel):
    def __init__(self,
            in_channels,
            latent_dim,
            hidden_dims,
            n_residual_blocks, 
            in_width,
            in_height,
            device,
            **kwargs) -> None:     

        super(BasicVAE, self).__init__() 
        self.in_channels = in_channels  # Image channels (5 in this case)
        self.latent_dim = latent_dim   
        self.hidden_dims = hidden_dims
        self.n_residual_blocks = n_residual_blocks
        self.in_width = in_width
        self.in_height = in_height
        self.device = device

        # Setup metrics 
        self.metrics = TrainingMetrics(self.in_height, self.in_width, self.in_channels, self.latent_dim, device = self.device)
        
        self.encoder = Encoder(
            in_channels = self.in_channels,
            latent_dim = self.latent_dim,
            hidden_dims = self.hidden_dims,
            n_residual_blocks = self.n_residual_blocks, 
            in_width = self.in_width,
            in_height = self.in_height,
        )

        self.decoder = Decoder(
            out_channel = self.in_channels,
            latent_dim = self.latent_dim,
            hidden_dims = self.hidden_dims,
            n_residual_blocks = self.n_residual_blocks, 
            out_width = self.in_width,
            out_height = self.in_height,
        ) 

    # Forward pass    
    def forward(self, X):
        mu, log_sigma = self.encoder(X)
        # Apply reparametrization trick
        z = self.reparameterize(mu, log_sigma)
        out = self.decoder(z)
        loss, recon_loss, kld = self.loss_function(X, out, mu, log_sigma).values()
        return  dict(out=out, z=z, loss=loss, recon_loss=recon_loss, kld=-kld)
        
    def reparameterize(self, mu, log_sigma, **kwargs):
        """
        Perform the reparametrization trick to allow for gradient descent. 
        mu: the mean of the latent space as predicted by the encoder module
        log_sigma: log variance of the latent space as predicted by the encoder 
        """
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def sample(self,
               num_samples:int, temperature:int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        num_samples: Number of samples
        device: Device to run the model
        """
        # Sample random vector
        z = torch.randn(num_samples,
                        self.latent_dim)*temperature
        z = z.to(self.device)
        samples = self.decoder(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        x: input image
        """
        return self.forward(x)['out']
    
    def get_latent_representation(self, X, **kwargs):
        """
        Given an input X, it returns a latent encoding for it 
        """
        mu, log_sigma = self.encoder(X)
        return dict(z=self.reparameterize(mu, log_sigma), mu=mu, log_sigma=log_sigma)

    def reconstruction_loss(self):
        pass

    def loss_function(self):
        pass
        
    def update_model(self, train_loader, epoch, optimizer):
        """
        Compute a forward step and returns the losses 
        """
        training_loss = 0
        tot_recons_loss = 0 
        tot_kl_loss = 0  

        self.metrics.reset()
        for batch in tqdm(train_loader):
            X_batch = batch['X'].to(self.device) # Load batch
            res = self.forward(X_batch)  # Predict and compute the loss on the model
            out, z, loss, recon_loss, kl_loss = res.values()  # Collect the losses 

            training_loss += loss.item()
            tot_recons_loss += recon_loss.item()
            tot_kl_loss += kl_loss.item()

            # Optimizer step  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.metrics.update_rmse(X_batch, out)

        avg_loss = training_loss/len(train_loader)
        avg_recon_loss = tot_recons_loss/len(train_loader)
        self.metrics.update_bpd(avg_recon_loss)
        avg_kl_loss = tot_kl_loss/len(train_loader)
        print(f'Mean loss after epoch {epoch}: {avg_loss}')
        print(f'Mean reconstruction loss after epoch {epoch}: {avg_recon_loss}')
        print(f'Mean kl divergence after epoch {epoch}: {avg_kl_loss}')
        self.metrics.print_metrics()
        return dict(loss=avg_loss, recon_loss=avg_loss, kl_loss=avg_kl_loss), self.metrics.metrics

    
    def evaluate(self, loader, dataset, checkpoint_path='', fold = 'val'):
        """
        Validation loop 
        """
        if fold == 'test':
            # Testing phase 
            self.load_state_dict(torch.load(checkpoint_path))

        val_loss = 0
        val_recon_loss = 0 
        val_kl_loss = 0 
        # Zero out the metrics for the next step
        self.metrics.reset()

        for val_batch in loader:
            X_val_batch = val_batch['X'].to(self.device) # Load batch
            with torch.no_grad():
                val_res = self.forward(X_val_batch)
            # Gather components of the output
            out_val, z, loss_val, recon_loss, kld_loss = val_res.values()
            self.metrics.update_rmse(X_val_batch, out_val)

            # Accumulate the validation loss 
            val_loss += loss_val.item()
            val_recon_loss += recon_loss.item()
            val_kl_loss += kld_loss.item()
        
        # Print loss results
        avg_validation_loss = val_loss/len(loader)
        avg_validation_recon_loss = val_recon_loss/len(loader)
        self.metrics.update_bpd(avg_validation_recon_loss)
        avg_validation_kld_loss = val_kl_loss/len(loader)
        print(f'Average validation loss: {avg_validation_loss}')
        print(f'Average validation reconstruction loss: {avg_validation_recon_loss}')
        print(f'Average kld reconstruction loss: {avg_validation_kld_loss}')
        self.metrics.print_metrics()
        return dict(loss=avg_validation_loss, recon_loss=avg_validation_recon_loss, kld_loss=avg_validation_kld_loss), self.metrics.metrics
        

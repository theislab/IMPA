import torch
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm
import numpy as np

def gaussian_nll(mu, log_sigma, x):
    """
    Implement Gaussian nll loss
    """
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

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

    def forward(self, x):
        out = self.resblock(x)
        out += x  # Residual connection 
        out = self.activation_out(out)
        return out


class Encoder(nn.Module):
    """
    The encoder module
    """
    def __init__(self,
                in_channels: int = 5,
                latent_dim: int = 128,
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

        # Convolutional structure
        self.modules = []
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128]
        
        # Downsizing convolutional blocks 
        for h_dim in self.hidden_dims:
            self.modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                                kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        # Additional convolutional module   
        self.modules.append(
            nn.Sequential(
                nn.Conv2d(self.hidden_dims[-1], self.hidden_dims[-1],
                            kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU())
                )    
        
        # Add residual blocks 
        for _ in range(self.n_residual_blocks):
            self.modules.append(ResidualLayer(self.hidden_dims[-1], self.hidden_dims[-1]))

        self.encoder = nn.Sequential(*self.modules)

        # Add bottleneck
        downsampling_factor_width = self.in_width//2**(len(self.hidden_dims) + 1)
        downsampling_factor_height = self.in_height//2**(len(self.hidden_dims) + 1)
        self.flattened_dim = self.hidden_dims[-1]*downsampling_factor_width*downsampling_factor_height
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)  # Mean encoding
        self.fc_var = nn.Linear(self.flattened_dim, latent_dim)  # Log-var encodings 

    def forward(self, X):
        X = self.encoder(X)  # Encode the image 
        X = self.flatten(X)   
        # Derive the encodings for the mean and the log variance
        mu = self.fc_mu(X)  
        log_sigma = self.fc_var(X)
        return [mu, log_sigma]


class Decoder(nn.Module):
    """
    The decoder module
    """
    def __init__(self,
                out_channels: int = 5,
                latent_dim: int = 128,
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

        # Build convolutional dimensions
        self.modules = []
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.hidden_dims = hidden_dims

        # Layer to upscale latent sample 
        self.upsampling_factor_width = self.out_width//2**(len(self.hidden_dims) + 1)
        self.upsampling_factor_height = self.out_height//2**(len(self.hidden_dims) + 1)
        self.flattened_dim = self.hidden_dims[0]*self.upsampling_factor_width*self.upsampling_factor_height
        self.upsample_fc = nn.Linear(self.latent_dim, self.flattened_dim)

        # Append the residual blocks
        for _ in range(self.n_residual_blocks):
            self.modules.append(ResidualLayer(self.hidden_dims[0], self.hidden_dims[0]))

        # Additional convolutional module   
        self.modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dims[0], self.hidden_dims[0],
                            kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU())
                )          

        # Add the final upsampling Conv2DTranspose network
        for i in range(len(self.hidden_dims) - 1):
            self.modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2),
                    nn.LeakyReLU())
            )

        # Extra layer activated with Tanh
        self.modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dims[-1],
                                   out_channels=self.out_channels,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.Sigmoid()))

        self.decoder = nn.Sequential(*self.modules)
    
    def forward(self, z):
        X = self.upsample_fc(z)
        # Reshape to height x width
        X = X.view(-1, self.hidden_dims[0], self.upsampling_factor_width, self.upsampling_factor_height)
        X = self.decoder(X)
        return X 


class ResidConvVAE(nn.Module):
    """
    Basic convolutional autoencoder with resnet blocks 
    """
    def __init__(self,
                in_channels: int = 5,
                latent_dim: int = 128,
                hidden_dims: list = None,
                n_residual_blocks: int = 6, 
                in_width: int = 64,
                in_height: int = 64,
                **kwargs) -> None:
        super(ResidConvVAE, self).__init__()      

        self.in_channels = in_channels  # Image channels (5 in this case)
        self.latent_dim = latent_dim  
        self.hidden_dims = hidden_dims
        self.n_residual_blocks = n_residual_blocks
        self.in_width = in_width
        self.in_height = in_height

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

        # Log-scale for the likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        
    def forward(self, X):
        mu, log_sigma = self.encoder(X)
        # Apply reparametrization trick
        z = self.reparameterize(mu, log_sigma)
        out = self.decoder(z)
        loss, recon_loss, kld = self.loss_function(X, out, mu, log_sigma).values()
        return  dict(out=out, loss=loss, recon_loss=recon_loss, kld=kld)


    def reparameterize(self, mu, log_sigma, **kwargs):
        """
        Perform the reparametrization trick to allow for gradient descent. 
        mu: the mean of the latent space as predicted by the encoder module
        log_sigma: log variance of the latent space as predicted by the encoder 
        """
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def reconstruction_loss(self, x_hat, x):
        """ 
        Computes the likelihood of the data given the latent variable,
        in this case using a Gaussian distribution with mean predicted by the neural network and variance = 1 
        """
        # Gaussian log lik
        #rec = gaussian_nll(x_hat, 0, x).sum((1,2,3)).mean()  # Single value (not averaged across batch element)
        rec = gaussian_nll(x_hat, self.log_scale, x).sum((1,2,3)).mean()
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
        recons_loss = gaussian_nll(X_hat, self.log_scale, X).sum((1,2,3)).mean()

        # KL divergence 
        #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu ** 2 - log_sigma.exp(), dim = 1), dim = 0)  # Canonical form of the analytic KL but inverted in sign 
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim=1), dim=0)
        
        # Total loss 
        loss = recons_loss + kld_loss
        # Zero out the gradient of the losses 
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}
        
    
    def sample(self,
               num_samples:int,
               device: int, seed:int=42, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        num_samples: Number of samples
        device: Device to run the model
        """
        torch.manual_seed(seed)
        # Sample random vector
        z = torch.randn(num_samples,
                        self.latent_dim)
        z = z.to(device)
        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        x: input image
        """
        return self.forward(x)['out']


    def update_model(self, train_loader, epoch, optimizer, device):
        """
        Compute a forward step and returns the losses 
        """
        training_loss = 0
        tot_recons_loss = 0 
        tot_kl_loss = 0  
        for batch in tqdm(train_loader): 
            batch = batch.to(device) # Load batch
            res = self.forward(batch)  # Predict and compute the loss on the model
            out, loss, recon_loss, kl_loss = res.values()  # Collect the losses 
            training_loss += loss.item()
            tot_recons_loss += recon_loss.item()
            tot_kl_loss += kl_loss.item()

            # Optimizer step  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = training_loss/len(train_loader)
        avg_recon_loss = tot_recons_loss/len(train_loader)
        avg_kl_loss = tot_kl_loss/len(train_loader)
        print(f'Mean loss after epoch {epoch}: {avg_loss}')
        print(f'Mean reconstruction loss after epoch {epoch}: {avg_recon_loss}')
        print(f'Mean kl divergence after epoch {epoch}: {avg_kl_loss}')
        return dict(loss=avg_loss, recon_loss=avg_loss, kl_loss=avg_kl_loss)

    
    def evaluate(self, loader, dataset, device, checkpoint_path='', fold = 'val'):
        """
        Validation loop 
        """
        if fold == 'test':
            # Testing phase 
            self.load_state_dict(torch.load(checkpoint_path))

        val_loss = 0
        val_recon_loss = 0 
        val_kl_loss = 0 
        for val_batch in loader:
            val_batch = val_batch.to(device) # Load batch
            with torch.no_grad():
                val_res = self.forward(val_batch)
            # Gather components of the output
            out_val, loss_val, recon_loss, kld_loss = val_res.values()

            # Accumulate the validation loss 
            val_loss += loss_val.item()
            val_recon_loss += recon_loss
            val_kl_loss += kld_loss
        
        avg_validation_loss = val_loss/len(loader)
        avg_validation_recon_loss = val_recon_loss/len(loader)
        avg_validation_kld_loss = val_kl_loss/len(loader)
        print(f'Average validation loss: {avg_validation_loss}')
        print(f'Average validation reconstruction loss: {avg_validation_recon_loss}')
        print(f'Average kld reconstruction loss: {avg_validation_kld_loss}')
        return dict(loss=avg_validation_loss, recon_loss=avg_validation_recon_loss, kld_loss=avg_validation_kld_loss)
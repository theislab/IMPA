import torch
import torch.utils.data
from torch import nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.models as models

def gaussian_nll(mu, log_sigma, x):
    """
    Implement Gaussian nll loss
    """
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor

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


class SigmaVAE(nn.Module):
    def __init__(self,
                in_channels: int = 5,
                latent_dim: int = 128,
                hidden_dims: list = None,
                n_residual_blocks: int = 6, 
                in_width: int = 64,
                in_height: int = 64,
                device = 'cuda',
                resnet = True,
                **kwargs) -> None:
        super(SigmaVAE, self).__init__()      

        self.device = device
        self.in_channels = in_channels  # Image channels (5 in this case)
        self.latent_dim = latent_dim   
        self.n_residual_blocks = n_residual_blocks
        self.in_width = in_width
        self.in_height = in_height
        self.resnet = resnet

        if hidden_dims is None:
            hidden_dims = 32
        self.hidden_dims = hidden_dims

        ## Build encoder network
        self.encoder = self.get_encoder(self.in_channels, self.hidden_dims, self.resnet, self.n_residual_blocks)

        # Obtain the size of the flattened input 
        demo_input = torch.ones([1, self.in_channels, self.in_width, self.in_height])
        h_dim = self.encoder(demo_input).shape[1]
        print('h_dim', h_dim)
        
        # Fully connected networks to latent space
        self.fc11 = nn.Linear(h_dim, self.latent_dim)
        self.fc12 = nn.Linear(h_dim, self.latent_dim)

        # Bilid the decoder 
        self.fc2 = nn.Linear(self.latent_dim, h_dim)
        self.decoder = self.get_decoder(self.hidden_dims, self.in_channels, self.resnet, self.n_residual_blocks)
        
        # The log sigma as an external parameter 
        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0.0)[0], requires_grad=True)
        
    @staticmethod
    def get_encoder(img_channels, filters_m, resnet, n_residual_blocks):
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(img_channels, filters_m, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(filters_m),
            nn.ReLU(),
            nn.Conv2d(filters_m, 2 * filters_m, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * filters_m, 4 * filters_m, (5, 5), stride=2, padding=2),
            nn.ReLU()))
        if resnet:
            for res in range(n_residual_blocks):
                modules.append(ResidualLayer(4 * filters_m, 4 * filters_m))
        modules.append(Flatten())
        return nn.Sequential(*modules)

    @staticmethod
    def get_decoder(filters_m, out_channels, resnet, n_residual_blocks):
        modules = [UnFlatten(4 * filters_m)]
        if resnet:
            for res in range(n_residual_blocks-1):
                modules.append(ResidualLayer(4 * filters_m, 4 * filters_m))
        modules.append(nn.Sequential(nn.ConvTranspose2d(4 * filters_m, 2 * filters_m, (6, 6), stride=2, padding=2),
            nn.BatchNorm2d(2*filters_m),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * filters_m, filters_m, (6, 6), stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(filters_m, out_channels, (5, 5), stride=1, padding=2),
            nn.Sigmoid()))
        return nn.Sequential(*modules)

    def encode(self, x):
        """
        Run encoder on an input 
        """
        h = self.encoder(x)
        return self.fc11(h), self.fc12(h)
    
    def reparameterize(self, mu, logvar):
        """
        VAE rparametrization trick 
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Run decoder on a latent embedding
        """
        return self.decoder(self.fc2(z))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        loss, recon_loss, kld = self.loss_function(x_hat, x, mu, logvar)
        return dict(out=x_hat, loss=loss, recon_loss=recon_loss, kld=kld)
    
    def sample(self, n):
        sample = torch.randn(n, self.latent_dim).to(self.device)
        return self.decode(sample)

    def reconstruction_loss(self, x_hat, x):
        """ Computes the likelihood of the data given the latent variable,
        in this case using a Gaussian distribution with mean predicted by the neural network and variance = 1 """
        # log rmse
        log_sigma = ((x - x_hat) ** 2).mean([0,1,2,3], keepdim=True).sqrt().log()  # Keep the 3 dimensions 
        self.log_sigma = torch.nn.Parameter(log_sigma)
        # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
        # ensures stable training.
        log_sigma = softclip(log_sigma, -6)
        # Gaussian log lik
        rec = gaussian_nll(x_hat, log_sigma, x).sum((1,2,3)).mean()  # Single value (not averaged across batch element)
        return rec

    def loss_function(self, recon_x, x, mu, logvar):
        """
        Aggregate the reconstruction and kl losses 
        """
        rec = self.reconstruction_loss(recon_x, x)
        # kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim = 0)
        return rec + kl, rec.detach(), kl.detach()

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



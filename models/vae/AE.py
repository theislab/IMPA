import torch
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm
from metrics import *


class AE(nn.Module):
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
        super(AE, self).__init__()      

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, X):
        z = self.encoder(X)
        # Apply reparametrization trick
        out = self.decoder(z)
        loss = self.loss_function(X, out)
        return  dict(out=out, loss=loss)

    
    def loss_function(self, X, X_hat) -> dict:
        """
        Computes the VAE loss 
        X: The input data 
        X_hat: The reconstucted input 
        mu: the mean encodings
        log_sigma: the log variance encodings 
        """
        # Reconstruction loss between the input and the prediction
        recons_loss = F.mse_loss(X, X_hat)
        # Zero out the gradient of the losses 
        return recons_loss

    
    def generate(self, X, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        x: input image
        """
        return self.forward(X)['out']

    
    def update_model(self, train_loader, epoch, optimizer, device):
        """
        Compute a forward step and returns the losses 
        """
        training_loss = 0 
        for batch in tqdm(train_loader): 
            batch = batch.to(device) # Load batch
            res = self.forward(batch)  # Predict and compute the loss on the model
            out, loss = res.values()  # Collect the losses 
            training_loss += loss.item()

            # Optimizer step  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = training_loss/len(train_loader)
        print(f'Total loss after epoch {epoch}: {avg_loss}')
        return dict(loss=avg_loss)

    
    def evaluate(self, loader, dataset, device, checkpoint_path='', fold = 'val'):
        """
        Validation loop 
        """
        if fold == 'test':
            # Testing phase 
            self.load_state_dict(torch.load(checkpoint_path))

        val_loss = 0
        # Zero out the metrics for the next step
        self.metrics.reset()

        for val_batch in loader:
            val_batch = val_batch.to(device) # Load batch
            with torch.no_grad():
                val_res = self.forward(val_batch)
            # Gather components of the output
            out_val, loss_val = val_res.values()
            self.metrics.metrics_update(out_val, val_batch)

            # Accumulate the validation loss 
            val_loss += loss_val.item()
        
        avg_validation_loss = val_loss/len(loader)
        print(f'Total validation loss: {avg_validation_loss}')
        self.metrics.print_metrics()
        return dict(loss=avg_validation_loss), self.metrics.print_metrics()


        




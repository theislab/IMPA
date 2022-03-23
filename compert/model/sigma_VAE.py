import torch
import torch.utils.data

from .CPA import *


class SigmaVAE(CPA):
    def __init__(self,
                in_width: int,
                in_height: int,
                in_channels: int,
                device: str,
                num_drugs: int,
                n_seen_drugs: int,
                seed: int = 0,
                patience: int = 5,
                hparams="",
                predict_n_cells=False,
                append_layer_width=None,
                drug_embeddings = None) -> None:
     
        super(SigmaVAE, self).__init__(in_width,
                                        in_height,
                                        in_channels,
                                        device,
                                        num_drugs,
                                        n_seen_drugs,   
                                        seed,
                                        patience,
                                        hparams,
                                        predict_n_cells,
                                        append_layer_width,
                                        drug_embeddings, 
                                        variational=True)


    def ae_loss(self, X, X_hat, mu, log_sigma):
        """
        Aggregate the reconstruction and kl losses 
        """
        rec = self.reconstruction_loss(X_hat, X)
        kl = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim = 1), dim = 0)
        loss = rec + kl
        return {'total_loss': loss, 'reconstruction_loss': rec.detach(), 'KLD': kl.detach()}

    
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
                        self.hparams["latent_dim"])*temperature
        z = z.to(self.device)
        samples = self.decoder(z)
        return samples

    
    def get_latent_representation(self, X):
        """
        Given an input X, it returns a latent encoding for it 
        """
        mu, log_sigma = self.encoder(X)
        return dict(z=self.reparameterize(mu, log_sigma), mu=mu, log_sigma=log_sigma)
    
    
    def generate(self, loader):
        """
        Given an input image x, returns the reconstructed image
        x: input image
        """
        original = next(iter(loader))
        original_X = original['X'][0].to(self.device).unsqueeze(0)
        original_id  = original['smile_id'][0].to(self.device).unsqueeze(0)
        original_emb = self.drug_embeddings(original_id)
        
        with torch.no_grad():
            mu_orig, log_sigma_orig = self.encoder(original_X)  # Encode image
            z_x = self.reparameterize(mu_orig, log_sigma_orig)  # Reparametrization trick 
            if self.adversarial:
                z_emb = self.drug_embedding_encoder(original_emb)
                z = z_x + z_emb
                reconstructed_X = self.decoder(z) 
            else:
                reconstructed_X = self.decoder(z_x) 
        return original_X, reconstructed_X

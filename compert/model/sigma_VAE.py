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
                append_layer_width=None,
                drug_embeddings = None, 
                dataset_name='cellpainting',
                predict_moa=False,
                n_moa=0, 
                total_iterations=None,
                class_weights: dict = None) -> None:
     
        super(SigmaVAE, self).__init__(in_width,
                                        in_height,
                                        in_channels,
                                        device,
                                        num_drugs,
                                        n_seen_drugs,   
                                        seed,
                                        patience,
                                        hparams,
                                        append_layer_width,
                                        drug_embeddings, 
                                        variational=True,
                                        dataset_name=dataset_name,
                                        predict_moa=predict_moa,
                                        n_moa=n_moa, 
                                        total_iterations=total_iterations, 
                                        class_weights = class_weights)


    def reconstruction_loss(self, X_hat, X):
        """ Computes the likelihood of the data given the latent variable,
        in this case using a Gaussian distribution with mean predicted by the neural network and variance = 1 (
        same for VAE and AE) """
        # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
        # ensures stable training.
        if self.hparams["data_driven_sigma"]:
            self.log_scale = ((X - X_hat) ** 2).mean([0,1,2,3], keepdim=True).sqrt().log()  # Keep the 3 dimensions 
            self.log_scale = softclip(self.log_scale, -6)
        # Gaussian log lik
        if self.hparams["mean_recon_loss"]:
            rec = gaussian_nll(X_hat, self.log_scale, X).mean()  # Single value (not averaged across batch element)
        else:
            rec = gaussian_nll(X_hat, self.log_scale, X).sum((1,2,3)).mean()  # Single value (not averaged across batch element)
        return rec

    
    def kl_loss(self, mu, log_sigma):
        """Compute KL divergence with a standard normal distribution

        Args:
            mu (torch.tensor): mean tensor
            log_sigma (torch.tensor): log sigma tensor
        """
        if self.hparams["mean_recon_loss"]:
            kl = torch.mean(-0.5 * torch.mean(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim = 1), dim = 0)
        else:
            kl = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim = 1), dim = 0)
        return kl


    def ae_loss(self, X, X_hat, mu, log_sigma):
        """
        Aggregate the reconstruction and kl losses 
        """
        rec = self.reconstruction_loss(X_hat, X)
        kl = self.kl_loss(mu, log_sigma)
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
        temperature: Temperature factor for sampling 
        """
        # TODO: give the chance to add the drug of interest by ID

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
        # Collect the data from the batch at random 
        original = next(iter(loader))
        original_X = original['X'][0].to(self.device).unsqueeze(0)  

        with torch.no_grad():
            mu_orig, log_sigma_orig = self.encoder(original_X) # Encode image
            z_basal = self.reparameterize(mu_orig, log_sigma_orig)  # Reparametrization trick 
            # Handle the case training is not adversarial 
            if not self.adversarial:
                reconstructed_X = self.decoder(z_basal) 

            else:
                # Collect the encoders for the drug embeddings to condition the latent space 
                drug_id  = original['smile_id'][0].to(self.device).unsqueeze(0)
                drug_emb = self.drug_embeddings(drug_id)
                z_drug = self.drug_embedding_encoder(drug_emb) 
                # Collect the mode of action embeddings 
                if self.predict_moa:
                    moa_id  = original['moa_id'][0].to(self.device).unsqueeze(0)
                    moa_emb = self.moa_embeddings(moa_id) 
                    z_moa = self.moa_embedding_encoder(moa_emb) 
                else:
                    z_moa = 0 
                
                # If not concat, perform the sum of embeddings 
                z = z_basal + z_drug + z_moa
                reconstructed_X = self.decoder(z) 

        return original_X, reconstructed_X

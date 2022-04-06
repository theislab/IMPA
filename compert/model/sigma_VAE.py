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
                n_moa=0) -> None:
     
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
                                        n_moa=n_moa)


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
        temperature: Temperature factor for sampling 
        """
        # TODO: give the chance to add the drug of interest by ID

        # Sample random vector
        z = torch.randn(num_samples,
                self.hparams["latent_dim"])*temperature

        # Concatenate with zeros - Choose the number of dimensions depending on the kind of concatenation performed 
        if self.hparams["concat_embedding"]:
            drug_dim = self.n_seen_drugs if self.hparams["concat_one_hot"] else self.hparams["drug_embedding_dimension"]
            z = torch.cat([z, torch.zeros(z.shape[0], drug_dim).to(self.device)], dim = 1)
            if self.predict_moa:
                moa_dim = self.n_moa if self.hparams["concat_one_hot"] else self.hparams["moa_embedding_dimension"]
                z = torch.cat([z, torch.zeros(z.shape[0], moa_dim).to(self.device)], dim = 1)
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

        # Collect the encoders for the drug embeddings to condition the latent space 
        drug_id  = original['smile_id'][0].to(self.device).unsqueeze(0)
        drug_emb = self.drug_embeddings(drug_id) 
        z_drug = self.drug_embedding_encoder(drug_emb) if not self.hparams["concat_one_hot"] else original["drug_one_hot"][0].to(self.device).unsqueeze(0)
        # Collect the mode of action embeddings 
        if self.predict_moa:
            moa_id  = original['moa_id'][0].to(self.device).unsqueeze(0)
            moa_emb = self.drug_embeddings(moa_id) 
            z_moa = self.moa_embedding_encoder(moa_emb) if not self.hparams["concat_one_hot"] else original["moa_one_hot"][0].to(self.device).unsqueeze(0)
        else:
            z_moa = 0 

        with torch.no_grad():
            mu_orig, log_sigma_orig = self.encoder(original_X)  # Encode image
            z_x = self.reparameterize(mu_orig, log_sigma_orig)  # Reparametrization trick 
            # Handle the case training is not adversarial 
            if not self.adversarial:
                if self.hparams["concat_embedding"]:
                    # The concatenation dimension is equal to the number of drugs if the one hot encoding is carried out 
                    drug_dim = self.n_seen_drugs if self.hparams["concat_one_hot"] else self.hparams["drug_embedding_dimension"]
                    z = torch.cat([z, torch.zeros(z.shape[0], drug_dim).to(self.device)], dim = 1)
                    if self.predict_moa:
                        moa_dim = self.n_moa if self.hparams["concat_one_hot"] else self.hparams["moa_embedding_dimension"]
                        z = torch.cat([z, torch.zeros(z.shape[0], moa_dim).to(self.device)], dim = 1)
                reconstructed_X = self.decoder(z_x) 

            else:
                # If not concat, perform the sum of embeddings 
                if not self.hparams["concat_embeddings"]:
                    z = z_x + z_drug + z_moa
                else:
                    z = torch.cat([z, z_drug], dim = 1)
                    if self.predict_moa:
                        z = torch.cat([z, z_moa], dim = 1) 
                reconstructed_X = self.decoder(z) 

        return original_X, reconstructed_X

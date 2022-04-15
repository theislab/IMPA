import torch
import torch.utils.data

from .CPA import *


class SigmaAE(CPA):
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
     
        super(SigmaAE, self).__init__(in_width,
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
                                        variational=False,
                                        dataset_name=dataset_name,
                                        predict_moa=predict_moa,
                                        n_moa=n_moa,
                                        total_iterations=total_iterations,
                                        class_weights=class_weights)

        
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

    def ae_loss(self, X, X_hat):
        """
        Aggregate and return the reconstruction loss
        """
        loss = self.reconstruction_loss(X_hat, X)        
        return dict(total_loss=loss)

    
    def get_latent_representation(self, X):
        """
        Given an input X, it returns a latent encoding for it 
        """
        z = self.encoder(X)
        return dict(z=z)
    

    def generate(self, loader):
        """
        Given an input image x, returns the reconstructed image
        x: input image
        """
        # Collect the data from the batch at random 
        original = next(iter(loader))
        original_X = original['X'][0].to(self.device).unsqueeze(0)  

        with torch.no_grad():
            z_basal = self.encoder(original_X)  # Encode image 
            # Handle the case training is not adversarial 
            if not self.adversarial:
                reconstructed_X = self.decoder(z_basal) 

            else:
                # Collect the encoders for the drug embeddings to condition the latent space 
                drug_id  = original['smile_id'][0].to(self.device).unsqueeze(0)
                drug_emb = self.drug_embeddings(drug_id) if not self.hparams["concat_one_hot"] else None
                z_drug = self.drug_embedding_encoder(drug_emb) if not self.hparams["concat_one_hot"] else original["mol_one_hot"][0].to(self.device).unsqueeze(0).float()
                # Collect the mode of action embeddings 
                if self.predict_moa:
                    moa_id  = original['moa_id'][0].to(self.device).unsqueeze(0)
                    moa_emb = self.moa_embeddings(moa_id) if not self.hparams["concat_one_hot"] else None
                    z_moa = self.moa_embedding_encoder(moa_emb) if not self.hparams["concat_one_hot"] else original["moa_one_hot"][0].to(self.device).unsqueeze(0).float()
                else:
                    z_moa = 0 
                
                # If not concat, perform the sum of embeddings 
                z = z_basal + z_drug + z_moa
                reconstructed_X = self.decoder(z) 

        return original_X, reconstructed_X
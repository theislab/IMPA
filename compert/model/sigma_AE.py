import torch
import torch.utils.data

from compert.training_utils import *
from compert.model.CPA import *


class SigmaAE(CPA):
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
     
        super(SigmaAE, self).__init__(adversarial,
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
                                        drug_embeddings, 
                                        variational=False)

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
        Given a dataset, sample an image and return its reconstruction
        """
        original = next(iter(loader))
        original_X = original['X'][0].to(self.device).unsqueeze(0)
        original_id  = original['smile_id'][0].to(self.device).unsqueeze(0)
        original_emb = self.drug_embeddings(original_id)
        
        with torch.no_grad():
            z_x = self.encoder(original_X)  # Encode image 
            if self.adversarial:
                z_emb = self.drug_embedding_encoder(original_emb)
                z = z_x + z_emb
                reconstructed_X = self.decoder(z) 
            else:
                reconstructed_X = self.decoder(z_x) 
        return original_X, reconstructed_X

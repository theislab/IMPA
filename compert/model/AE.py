import torch
import torch.utils.data
from .autoencoders import initialize_encoder_decoder
from .CPA import *


class AE(torch.nn.Module):
    def __init__(self, in_channels, in_width, in_height, variational, hparams, extra_fm, n_seen_drugs, device) -> None:
     
        super(AE, self).__init__()
        self.in_channels = in_channels
        self.in_width = in_width
        self.in_height = in_height
        self.hparams = hparams
        self.variational = variational
        self.extra_fm = extra_fm
        self.encoder, self.decoder  = initialize_encoder_decoder(self.in_channels, 
                                                                self.in_width, 
                                                                self.in_height, 
                                                                self.variational, 
                                                                self.hparams, 
                                                                self.extra_fm)
        self.n_seen_drugs = n_seen_drugs
        self.device = device
        # Beta is used to control the weight of the kl loss
        self.beta = self.hparams['beta']

    def kl_loss(self, mu, log_sigma):
        """Compute KL divergence with a standard normal distribution

        Args:
            mu (torch.tensor): mean tensor
            log_sigma (torch.tensor): log sigma tensor
        """
        dims = list(range(len(mu.shape)))  # Either for linear latent or for feature map latent 
        if self.hparams["mean_recon_loss"]:
            kl = torch.mean(-0.5 * torch.mean(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim = dims[1:]), dim = 0)
        else:
            kl = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim = dims[1:]), dim = 0)
        return kl

    def reparameterize(self, mu, log_sigma):
        """
        Perform the reparametrization trick to allow for gradient descent. 
        mu: the mean of the latent space as predicted by the encoder module
        log_sigma: log variance of the latent space as predicted by the encoder 
        """
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu
   
    def reconstruction_loss(self, X_hat, X):
        """ 
        Reconstruction loss is the L1 loss (better at preserving sharp details)
        """
        return torch.nn.L1Loss()(X_hat, X)


    def ae_loss(self, X, X_hat, mu=None, log_sigma=None):
        """
        Aggregate and return the reconstruction loss
        """
        rec = self.reconstruction_loss(X_hat, X)   
        if self.variational:
            kl = self.kl_loss(mu, log_sigma)
            loss = rec + self.beta*kl
            return dict(total_loss=loss, recon_loss=rec, kl_loss=kl)
        else:
            loss = rec
            return dict(total_loss=loss)


    def get_latent_representation(self, X):
        """
        Given an input X, it returns a latent encoding for it 
        """
        z = self.encoder(X)
        return dict(z=z)


    # Forward pass    
    def forward_ae(self, X, y_drug=None, mode='train'):
        """Simple autoencoder forward pass used both in train and validation mode. 

        Args:
            X (torch.Tensor): The input data of interest
            y_drug (torch.Tensor, optional): The label for the drug. Can be encoded or one-hot depending on conditioning type. Defaults to None.
            mode (str, optional): in what mode we run it (train/eval). Defaults to 'train'.
        """
        # FORWARD STEP ENCODER
        if not self.variational:
            z_basal = self.encoder(X)
        else:
            mu, log_sigma = self.encoder(X)
            z_basal = self.reparameterize(mu, log_sigma)  
                
        # DECODE THE BASAL STATE ONLY IN EVALUATION MODE
        if mode=='eval' or y_drug == None:
            if self.hparams['decoding_style'] == 'sum':
                cond_drug = torch.zeros_like(z_basal).to(self.device)

            elif (self.hparams['decoding_style'] == 'concat'and self.hparams['concatenate_one_hot']):
                cond_drug = torch.zeros(z_basal.shape[0], self.n_seen_drugs).to(self.device)
            
            else:
                cond_drug = torch.zeros(z_basal.shape[0], self.hparams["drug_embedding_dimension"]).to(self.device)
            
            # DECODE BASAL
            out_basal, z = self.decoder(z_basal, cond_drug)

        else:
            out_basal = None
    
        # IF y_drug IS DIFFERENT FROM NONE, WE CONDITION THE LATENT
        if y_drug != None:
            # DECODE CONDITIONED SPACE
            out, z = self.decoder(z_basal, y_drug) 
            
        else:
            out = out_basal
        
        # Compute autoencoder loss
        if self.variational:
            ae_loss = self.ae_loss(X, out, mu, log_sigma)
        else:
            ae_loss = self.ae_loss(X, out)
        
        return dict(out=out, out_basal=out_basal, z=z, z_basal=z_basal, ae_loss=ae_loss)

    
    def generate(self, loader, drug_embeddings, drug_embedding_encoder, adversarial):
        """
        Given an input image x, returns the reconstructed image
        x: input image
        """
        # COLLECT DATA FROM NEXT BATCH
        original = next(iter(loader))  
        original_X = original['X'][0].to(self.device).unsqueeze(0) 
        # GROUND TRUTH WITH ONE-HOT VECTORS
        y_drug = original['mol_one_hot'].to(self.device).float()
        # DRUG ID
        drug_id = y_drug.argmax(1).to(self.device)

        with torch.no_grad():
            # Handle the case training is not adversarial (append zero masks to the )
            if not adversarial:
                # From basal reconstruct the image if not adversarial     
                reconstructed_X, _, _, _, _ = self.forward_ae(original_X, None).values()

            if adversarial:
                # If not concat, perform the sum of embeddings 
                if self.hparams["decoding_style"] == 'sum' or (self.hparams["decoding_style"] == 'concat' and not self.hparams["concatenate_one_hot"]):
                    # Collect the encoders for the drug embeddings to condition the latent space 
                    drug_emb = drug_embeddings(drug_id) 
                    z_drug = drug_embedding_encoder(drug_emb) 
                    y_drug = z_drug
                            
                # Decode based on drug conditioning 
                reconstructed_X, _, _, _, _ = self.forward_ae(original_X, y_drug=y_drug).values()
        return original_X, reconstructed_X

import torch
import torch.utils.data
from .autoencoders import initialize_encoder_decoder
from .CPA import *


class AE(torch.nn.Module):
    def __init__(self, in_channels, in_width, in_height, variational, hparams, extra_fm, n_seen_drugs, n_moa, device) -> None:
     
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
        self.n_moa = n_moa
        self.device = device
        
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
            loss = rec + self.hparams['beta']*kl
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
    def forward_ae(self, X, y_drug=None, y_moa=None, mode='train'):
        """Simple autoencoder forward pass used both in train and validation mode. 

        Args:
            X (torch.Tensor): The input data of interest
            y_drug (torch.Tensor, optional): The label for the drug. Can be encoded or one-hot depending on conditioning type. Defaults to None.
            y_moa (torch.Tensor, optional): The label for the moa. Can be encoded or one-hot depending on conditioning type. Defaults to None.
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
                cond_drug = None
                cond_moa = None

            if (self.hparams['decoding_style'] == 'concat'and self.hparams['concatenate_one_hot']):
                cond_drug = torch.zeros(z_basal.shape[0], self.n_seen_drugs).to(self.device)
                cond_moa = torch.zeros(z_basal.shape[0], self.n_moa).to(self.device)
            
            else:
                cond_drug = torch.zeros(z_basal.shape[0], self.hparams["drug_embedding_dimension"]).to(self.device)
                cond_moa = torch.zeros(z_basal.shape[0], self.hparams["moa_embedding_dimension"]).to(self.device)
            
            # DECODE BASAL
            out_basal = self.decoder(z_basal, cond_drug, cond_moa)
        else:
            out_basal = None

        # IF y_drug IS DIFFERENT FROM NONE, WE CONDITION THE LATENT
        if y_drug != None:
            if self.hparams["decoding_style"] == 'sum':
                # Sum the latents of the drug and the image
                z = z_basal + y_drug + y_moa           
                y_drug, y_moa = None, None
            else:
                z = z_basal  # We will concatenate to it in the decoder part 
            
            # DECODE CONDITIONED SPACE
            out = self.decoder(z, y_drug, y_moa) 
        else:
            z = None
            out = out_basal
        
        # Compute autoencoder loss
        if self.variational:
            ae_loss = self.ae_loss(X, out, mu, log_sigma)
        else:
            ae_loss = self.ae_loss(X, out)
        
        return dict(out=out, out_basal=out_basal, z=z, z_basal=z_basal, ae_loss=ae_loss)

    
    def generate(self, loader, drug_embeddings, drug_embedding_encoder, predict_moa, moa_embeddings, moa_embedding_encoder, adversarial):
        """
        Given an input image x, returns the reconstructed image
        x: input image
        """
        # COLLECT DATA FROM NEXT BATCH
        original = next(iter(loader))  
        original_X = original['X'][0].to(self.device).unsqueeze(0) 
        # GROUND TRUTH WITH ONE-HOT VECTORS
        y_drug = original['mol_one_hot'].to(self.device).float()
        y_moa = original['moa_one_hot'].to(self.device).float()
        # DRUG AND MOA ID
        drug_id = y_drug.argmax(1).to(self.device)
        moa_id = y_moa.argmax(1).to(self.device)

        with torch.no_grad():
            if adversarial:
                # Collect the encoders for the drug embeddings to condition the latent space 
                drug_emb = drug_embeddings(drug_id) 
                z_drug = drug_embedding_encoder(drug_emb) 
                # Collect the mode of action embeddings 
                if predict_moa:
                    moa_emb = moa_embeddings(moa_id) 
                    z_moa = moa_embedding_encoder(moa_emb)
                else:
                    z_moa = 0 
            
            # Encode image 
            if not self.variational:
                z_basal = self.encoder(original_X)  
            else:
                mu_orig, log_sigma_orig = self.encoder(original_X) # Encode image
                z_basal = self.reparameterize(mu_orig, log_sigma_orig)  # Reparametrization trick 

            # Handle the case training is not adversarial (append zero masks to the )
            if not adversarial:
                if self.hparams["decoding_style"] == 'sum':
                    y_drug = None
                    y_moa = None
                elif self.hparams["decoding_style"] == 'concat' and self.hparams["concatenate_one_hot"]:
                    y_drug = torch.zeros(original_X.shape[0], self.n_seen_drugs).to(self.device)
                    y_moa = torch.zeros(original_X.shape[0], self.n_moa).to(self.device)
                else:
                    y_drug = torch.zeros(z_basal.shape[0], self.hparams["drug_embedding_dimension"]).to(self.device)
                    y_moa = torch.zeros(z_basal.shape[0], self.hparams["moa_embedding_dimension"]).to(self.device)
                
                # From basal reconstruct the image if not adversarial     
                reconstructed_X = self.decoder(z_basal, y_drug, y_moa) 

            else:
                # If not concat, perform the sum of embeddings 
                if self.hparams["decoding_style"] == 'sum':
                    z = z_basal + z_drug + z_moa
                # If concat, perform the embedding concatenation 
                else:
                    if not self.hparams["concatenate_one_hot"]:
                        y_drug = z_drug
                        y_moa = z_moa
                    z = z_basal
                reconstructed_X = self.decoder(z, y_drug, y_moa) 

        return original_X, reconstructed_X

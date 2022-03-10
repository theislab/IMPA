import torch
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm
import numpy as np

from compert.model.modules import *
from compert.model.sigma_VAE import *
from compert.model.template_model import *

from metrics.metrics import *


"""
The default AE/VAE class 
"""

class CPA(TemplateModel):
    def __init__(self,
            adversarial: bool,
            in_width: int,
            in_height: int,
            in_channels: int,
            device: str,
            num_drugs: int,
            n_seen_drugs:int,
            seed: int = 0,
            patience: int = 5,
            hparams="",
            binary_task=False,
            append_layer_width=None,
            drug_embeddings = None):     

        super(CPA, self).__init__() 
        self.adversarial = adversarial  # If train a normal VAE or and adversarial VAE s
        self.in_width = in_width  # Image width 
        self.in_height = in_height  # Image height
        self.in_channels = in_channels  # Image channels (5 in this case)
        self.device = device

        # Parameters for the adversarial training 
        self.num_drugs = num_drugs 
        self.n_seen_drugs = n_seen_drugs
        self.seed = seed
        self.binary_task = binary_task 
        self.append_layer_width = append_layer_width
        self.drug_embeddings = drug_embeddings

        # Parameters for early-stopping 
        self.best_score = -np.inf
        self.patience = patience
        self.patience_trials = 0

        # set hyperparameters
        if isinstance(hparams, dict):
            self.hparams = hparams
        else:
            self.set_hparams_(seed, hparams)

        # Setup metrics 
        self.metrics = TrainingMetrics(self.in_height, self.in_width, self.in_channels, self.hparams["latent_dim"], device = self.device)
        
        # Instantiate the convolutional encoder and decoder modules
        self.encoder = Encoder(
            in_channels = self.in_channels,
            latent_dim = self.hparams["latent_dim"],
            hidden_dims = self.hparams["hidden_dim"],
            n_residual_blocks = self.hparams["n_residual_blocks"], 
            in_width = self.in_width,
            in_height = self.in_height,
        )

        self.decoder = Decoder(
            out_channel = self.in_channels,
            latent_dim = self.hparams["latent_dim"],
            hidden_dims = self.hparams["hidden_dim"],
            n_residual_blocks = self.hparams["n_residual_blocks"],  
            out_width = self.in_width,
            out_height = self.in_height,
        ) 

        # The log sigma as an external parameter
        self.log_scale = torch.nn.Parameter(torch.full((1,1,1,1), 0.0), requires_grad=True)

        # Can train with adversarial loss or a normal VAE (use adversary = False in latter case)
        if not self.adversarial:
            # Setup the optimizer and scheduler for the standard VAE 
            self.optimizer_autoencoder = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams["autoencoder_lr"],
                weight_decay=self.hparams["autoencoder_wd"],
            )

            
            # Learning rate schedulers for the model 
            self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
                self.optimizer_autoencoder,
                step_size=self.hparams["step_size_lr"],
                gamma=0.5,
            )


        if self.adversarial:
            # Instantiate the drug embedding and adversarial part of the network 
            if self.num_drugs > 0:
                # Adversary network is a simple MLP with custom depth and width 
                if not self.binary_task:
                    self.adversary_drugs = MLP(
                        [self.hparams["latent_dim"]]
                        + [self.hparams["adversary_width"]] * self.hparams["adversary_depth"]
                        + [self.n_seen_drugs]
                    )
                else:
                    self.adversary_drugs = MLP(
                        [self.hparams["latent_dim"]]
                        + [self.hparams["adversary_width"]] * self.hparams["adversary_depth"]
                        + [1]
                    )

                # Set the drug embedding to a fixed or trainable status depending on whether a pre-trained model is used 
                if self.drug_embeddings is None:
                    self.drug_embeddings = torch.nn.Embedding(
                        self.num_drugs, self.hparams["latent_dim"]
                    )
                    embedding_requires_grad = True
                else:
                    self.drug_embeddings = self.drug_embeddings
                    embedding_requires_grad = False

                # Drug embedding encoder 
                self.drug_embedding_encoder = MLP(
                    [self.drug_embeddings.embedding_dim]
                    + [self.hparams["embedding_encoder_width"]]
                    * self.hparams["embedding_encoder_depth"]
                    + [self.hparams["latent_dim"]],
                    last_layer_act="linear",
                )

                # Binary task predicts active versus inactive using the specific drug 
                if self.binary_task:
                    self.loss_adversary_drugs = torch.nn.BCEWithLogitsLoss(reduction = 'mean')
                else:
                    self.loss_adversary_drugs = torch.nn.CrossEntropyLoss(reduction = 'mean')


            # Now initialize the optimizer and the parameters needed for backpropagation
            has_drugs = self.num_drugs > 0
            get_params = lambda model, cond: list(model.parameters()) if cond else []  # Get the parameters of a model if a condition is verified
            # Collect parameters 
            _parameters = (
                get_params(self.encoder, True)
                + get_params(self.decoder, True)
                + get_params(self.drug_embeddings, has_drugs and embedding_requires_grad)
                + get_params(self.drug_embedding_encoder, True)
            )

            # Optimizer for the autoencoder 
            self.optimizer_autoencoder = torch.optim.Adam(
                _parameters,
                lr=self.hparams["autoencoder_lr"],
                weight_decay=self.hparams["autoencoder_wd"],
            )

            # Optimizer for the drug adversary. Make sure that only the right parameters are bound to it
            _parameters = get_params(self.adversary_drugs, has_drugs)

            self.optimizer_adversaries = torch.optim.Adam(
                _parameters,
                lr=self.hparams["adversary_lr"],
                weight_decay=self.hparams["adversary_wd"],
            )

            # Learning rate schedulers for the model 
            self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
                self.optimizer_autoencoder,
                step_size=self.hparams["step_size_lr"],
                gamma=0.5,
            )

            self.scheduler_adversary = torch.optim.lr_scheduler.StepLR(
                self.optimizer_adversaries,
                step_size=self.hparams["step_size_lr"],
                gamma=0.5,
            )

        # Initialize warmup params
        self.warmup_steps = self.hparams["warmup_steps"]

        # Statistics history
        self.history = {"epoch": [], "stats_epoch": []}

        # Iterations to decide when to perform adversarial training 
        self.iteration = 0

    
    def set_hparams_(self, seed, hparams):
        """
        Set hyper-parameters to (i) default values if `seed=0`, (ii) random
        values if `seed != 0`, or (ii) values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """
        default = seed == 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.hparams = {
            "latent_dim": 512 if default else int(np.random.choice([256, 512, 1024])),
            "hidden_dim": 64 if default else int(np.random.choice([32, 64, 128])),
            "depth": 3 if default else int(np.random.choice([3, 4, 5])),
            "n_residual_blocks":12 if default else int(np.random.choice([6, 12, 18])),

            "adversary_width": 128 if default else int(np.random.choice([64, 128, 256])),
            "adversary_depth": 3 if default else int(np.random.choice([2, 3, 4])),
            "reg_adversary": 5 if default else float(10 ** np.random.uniform(-2, 2)),  # Regularization 
            "penalty_adversary": 3 if default else float(10 ** np.random.uniform(-2, 1)),
            
            "autoencoder_lr": 1e-4 if default else float(10 ** np.random.uniform(-5, -3)),
            "adversary_lr": 3e-4 if default else float(10 ** np.random.uniform(-5, -3)),
            "autoencoder_wd": 1e-6 if default else float(10 ** np.random.uniform(-8, -4)),
            "adversary_wd": 1e-4 if default else float(10 ** np.random.uniform(-6, -3)),
            "adversary_steps": 3 if default else int(np.random.choice([1, 2, 3, 4, 5])),
            "batch_size": 128 if default else int(np.random.choice([64, 128, 256, 512])),
            "step_size_lr": 45 if default else int(np.random.choice([15, 25, 45])),
            "embedding_encoder_width": 512,
            "embedding_encoder_depth": 0,
            "warmup_steps": 5 if default else int(np.random.choice([0, 5, 10, 15]))
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams

    # Forward pass    
    def forward_ae(self, X):
        """Simple encoding-decoding process

        Args:
            X (torch.Tensor): The image data X

        Returns:
            dict: The reconstructed input, the latent representation and the losses  
        """
        mu, log_sigma = self.encoder(X)
        # Apply reparametrization trick
        z = self.reparameterize(mu, log_sigma)
        out = self.decoder(z)
        loss, recon_loss, kld = self.vae_loss(X, out, mu, log_sigma).values()
        return  dict(out=out, z=z, loss=loss, recon_loss=recon_loss, kld=-kld)

    
    def forward_compert(self, X, y_adv, drug_ids):
        """The forward step with adversarial training

        Args:
            X (torch.Tensor): the image data X
            y_adv (torch.Tensor): the target for the adversarial training  
            drug_embedding (torch.Tensor): The pre-computed embeddings for the drugs in the batch
        """
        # Autoencoder pass
        mu, log_sigma = self.encoder(X)  # Encode image
        z_basal = self.reparameterize(mu, log_sigma)  # Reparametrization trick 
        y_adv_hat = self.adversary_drugs(z_basal)


        # Embed the drug
        drug_embedding = self.drug_embeddings(drug_ids)
        z_drug = self.drug_embedding_encoder(drug_embedding)  # Embed input drug

        # Sum the latents of the drug and the image 
        z_adv = z_basal + z_drug  
        out = self.decoder(z_adv) 

        # Compute the adversarial loss
        adv_loss = self.loss_adversary_drugs(y_adv_hat, torch.argmax(y_adv, dim = 1))

        # The autoencoder loss 
        ae_loss, recon_loss, kld = self.vae_loss(X, out, mu, log_sigma).values() 

        # Check whether to perform adversarial training 
        if self.training:
            if (self.iteration % self.hparams["adversary_steps"]) == 0:
                # Compute the gradient penalty for the drug regularization term 
                adv_drugs_grad_penalty = self.compute_gradient_penalty(y_adv_hat.sum(), z_basal)
                loss = adv_loss + self.hparams["penalty_adversary"] * adv_drugs_grad_penalty
            
            else:
                loss = ae_loss - self.hparams["reg_adversary"] * adv_loss
        else:
            loss = adv_loss
        
        return dict(out=out, z=z_basal, loss=loss, ae_loss=ae_loss, recon_loss=recon_loss, kld=kld, adv_loss=adv_loss)
    
    def evaluate(self, X, y_adv=None, drug_id=None):
        """Perform evaluation step

        Args:
            X (torch.tensor): The batch of observations
            y_adv (torch.tensor): The drug label of the observation 
            drug_embed (torch.tensor): The embedding of the drug 
        """
        with torch.no_grad():
            mu, log_sigma = self.encoder(X)
            if not self.adversarial:
                z = self.reparameterize(mu, log_sigma)
                out = self.decoder(z)
                ae_loss, recon_loss, kld = self.vae_loss(X, out, mu, log_sigma).values()
                return dict(out=out, z=z, ae_loss=ae_loss, recon_loss=recon_loss, kld=kld)

            else:
                z_basal = self.reparameterize(mu, log_sigma)
                y_hat = self.adversary_drugs(z_basal)
                drug_embedding = self.drug_embeddings(drug_id)
                z_drug = self.drug_embedding_encoder(drug_embedding)  # Embed input drug
                # Sum the latents of the drug and the image 
                z = z_basal + z_drug  
                out = self.decoder(z)
                ae_loss, recon_loss, kld = self.vae_loss(X, out, mu, log_sigma).values()
                return dict(out=out, z_basal=z_basal, z=z, y_hat=y_hat, ae_loss=ae_loss, recon_loss=recon_loss, kld=kld)
            
        
    def compute_gradient_penalty(self, output, input):
        """Compute the penalty of the gradient of an output with respect to an input tensor

        Args:
            output (_torch.nn.Tensor_): Result of a differentiable function 
            input (_torch.nn.Tensor_): The input

        Returns:
            torch.Tensor: The gradient penalty associated to the gradient of output with respect to input 
        """
        grads = torch.autograd.grad(output, input, create_graph=True)
        grads = grads[0].pow(2).mean()
        return grads


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
    

    def get_latent_representation(self, X, **kwargs):
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
            z_emb = self.drug_embedding_encoder(original_emb)
            z = z_x + z_emb
            reconstructed_X = self.decoder(z) 
        return original_X, reconstructed_X
    
    
    def early_stopping(self, score):
        """
        Possibly early-stops training.
        """
        cond = score > self.best_score
        if cond:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1
        return cond, self.patience_trials > self.patience


    def vae_loss(self):
        pass

    
    def update_model(self, train_loader, epoch):
        """
        Compute a forward step and returns the losses 
        """
        if self.adversarial:
            training_loss = 0
            tot_recons_loss = 0
            tot_kl_loss = 0
            tot_ae_loss = 0
            tot_adv_loss = 0
        else:
            training_loss = 0
            tot_recons_loss = 0
            tot_kl_loss = 0
        
        
        self.metrics.reset()
        for batch in tqdm(train_loader):
            # For printing purposes 
            step = 'reconstruction' if (self.iteration % self.hparams["adversary_steps"]) == 0 else 'discrimination'

            # Collect the data from the batch 
            X = batch['X'].to(self.device)

            # If no adversarial training, simply perform the VAE training loop
            if not self.adversarial:
                res = self.forward_ae(X)
                out, z, loss, recon_loss, kl_loss = res.values()  # Collect the losses 

                # Optimizer step 
                self.optimizer_autoencoder.zero_grad()
                loss.backward()
                self.optimizer_autoencoder.step()

            else:
                # self.binary is true when the task is predicting trt vs dmso 
                if self.binary_task:
                    y_adv = batch['state'].to(self.device).long()
                else:
                    y_adv = batch['mol_one_hot'].to(self.device).long()
                
                drug_id = batch["smile_id"].to(self.device)
                del batch # Free memory
                
                # Forward pass
                out, _, loss, ae_loss, recon_loss, kl_loss, adv_loss = self.forward_compert(X, y_adv, drug_id).values()

                tot_ae_loss += ae_loss.item()
                tot_adv_loss += adv_loss.item()
                
                # Optimizer step - non-adversarial training 
                if (self.iteration % self.hparams["adversary_steps"]) == 0:       
                    self.optimizer_adversaries.zero_grad()
                    loss.backward()
                    self.optimizer_adversaries.step()
                
                else:
                    self.optimizer_autoencoder.zero_grad()
                    loss.backward()
                    self.optimizer_autoencoder.step()
            

            # Increase number of iterations
            self.iteration += 1

            # Update the global losses 
            training_loss += loss.item()
            tot_recons_loss += recon_loss.item()
            tot_kl_loss += kl_loss.item()

            # Perform optimizer step depending on the iteration
            self.metrics.update_rmse(X, out)

        # Print the loss metrics 
        avg_loss = training_loss/len(train_loader)
        avg_recon_loss = tot_recons_loss/len(train_loader)
        self.metrics.update_bpd(avg_recon_loss)
        avg_kl_loss = tot_kl_loss/len(train_loader)

        print(f'Mean loss after epoch {epoch}: {avg_loss}')
        print(f'Mean reconstruction loss after epoch {epoch}: {avg_recon_loss}')
        print(f'Mean kl divergence after epoch {epoch}: {avg_kl_loss}')


        if self.adversarial:
            avg_ae_loss = tot_ae_loss/len(train_loader)
            avg_adv_loss = tot_adv_loss/len(train_loader)
            print(f'Mean autoencoder loss after epoch {epoch}: {avg_ae_loss}')
            print(f'Mean adversarial loss after epoch {epoch}: {tot_adv_loss}')
            self.metrics.print_metrics()
            return dict(loss=avg_loss, recon_loss=avg_recon_loss, kl_loss=avg_kl_loss, avg_ae_loss=avg_ae_loss, avg_adv_loss=avg_adv_loss), self.metrics.metrics

        else: 
            return dict(loss=avg_loss, recon_loss=avg_recon_loss, kl_loss=avg_kl_loss), self.metrics.metrics
        

        
    


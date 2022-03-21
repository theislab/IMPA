import json
from tqdm import tqdm
import numpy as np
import torch

from .modules import *
from .template_model import *

from ..metrics.metrics import *


"""
The default AE/VAE class 
"""

class CPA(TemplateModel):
    def __init__(self,
            in_width: int,
            in_height: int,
            in_channels: int,
            device: str,
            num_drugs: int,
            n_seen_drugs:int,
            seed: int = 0,
            patience: int = 5,
            hparams="",
            predict_n_cells=False,
            append_layer_width=None,
            drug_embeddings = None,
            variational: bool = True):     

        super(CPA, self).__init__() 
        self.adversarial = False  # If a normal model or an adversarial network must be trained
        self.in_width = in_width  # Image width 
        self.in_height = in_height  # Image height
        self.in_channels = in_channels  # Image channels (5 in this case)
        self.device = device

        # Parameters for the adversarial training 
        self.num_drugs = num_drugs 
        self.n_seen_drugs = n_seen_drugs
        self.seed = seed
        self.predict_n_cells = predict_n_cells 
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

        # Setup metrics object
        self.metrics = TrainingMetrics(self.in_height, self.in_width, self.in_channels, self.hparams["latent_dim"], device = self.device)

        # The log sigma as an external parameter of the sigma vae training paradigm 
        if self.hparams["data_driven_sigma"]:
            self.log_scale = 0
        else:
            self.log_scale = torch.nn.Parameter(torch.full((1,1,1,1), 0.0), requires_grad=True)
        self.variational = variational 

        # Instantiate the convolutional encoder and decoder modules
        self.encoder = Encoder(
            in_channels = self.in_channels,
            latent_dim = self.hparams["latent_dim"],
            init_fm = self.hparams["init_fm"],
            n_conv = self.hparams["n_conv"],
            n_residual_blocks = self.hparams["n_residual_blocks"], 
            in_width = self.in_width,
            in_height = self.in_height,
            variational = self.variational
        )

        self.decoder = Decoder(
            out_channels = self.in_channels,
            latent_dim = self.hparams["latent_dim"],
            init_fm = self.hparams["init_fm"],
            n_conv = self.hparams["n_conv"],
            n_residual_blocks = self.hparams["n_residual_blocks"],  
            out_width = self.in_width,
            out_height = self.in_height,
            variational = self.variational
        ) 

        # Initialize warmup params
        self.warmup_steps = self.hparams["warmup_steps"]

        # Statistics history
        self.history = {"train":{'epoch':[]}, "val":{'epoch':[]}, "test":{'epoch':[]}, "ood":{'epoch':[]}}

        # Iterations to decide when to perform adversarial training 
        self.iteration = 0

        # Initialize the autoencoder first with adversarial equal to False
        self.initialize_ae() 

    def initialize_ae(self):
        print('Initalize autoencoder')
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

    def initialize_adversarial(self):
        print('Initalize adversarial training')
        # Adversary network is a simple MLP with custom depth and width 
        if not self.predict_n_cells:
            self.adversary_drugs = MLP(
                [self.hparams["latent_dim"]]
                + [self.hparams["adversary_width"]] * self.hparams["adversary_depth"]
                + [self.n_seen_drugs]
            ).to(self.device)
        else:
            self.adversary_drugs = MLP(
                [self.hparams["latent_dim"]]
                + [self.hparams["adversary_width"]] * self.hparams["adversary_depth"]
                + [1]
            ).to(self.device)

        # Set the drug embedding to a fixed or trainable status depending on whether a pre-trained model is used 
        if self.drug_embeddings is None:
            self.drug_embeddings = torch.nn.Embedding(
                self.num_drugs, self.hparams["latent_dim"])
            embedding_requires_grad = True
        else:
            self.drug_embeddings = self.drug_embeddings  # From pre-trained 
            embedding_requires_grad = False

        # Drug embedding encoder 
        self.drug_embedding_encoder = MLP(
            [self.drug_embeddings.embedding_dim]
            + [self.hparams["embedding_encoder_width"]]
            * self.hparams["embedding_encoder_depth"]
            + [self.hparams["latent_dim"]],
            last_layer_act="linear",
        ).to(self.device)

        # Binary task predicts active versus inactive using the specific drug 
        if self.predict_n_cells:
            self.loss_adversary_drugs = torch.nn.MSELoss(reduction = 'mean')
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

        self.scheduler_adversaries = torch.optim.lr_scheduler.StepLR(
            self.optimizer_adversaries,
            step_size=self.hparams["step_size_lr"],
            gamma=0.5,
        )

        # Turn adversarial to True
        self.adversarial = True

    
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
            "init_fm": 64 if default else int(np.random.choice([32, 64, 128])),  # Will be upsampled depth times
            "n_conv": 3 if default else int(np.random.choice([3, 4, 5])),
            "n_residual_blocks":12 if default else int(np.random.choice([6, 12, 18])),

            "adversary_width": 128 if default else int(np.random.choice([64, 128, 256])),
            "adversary_depth": 3 if default else int(np.random.choice([2, 3, 4])),
            "reg_adversary": 5 if default else float(10 ** np.random.uniform(-2, 2)),  # Regularization 
            "penalty_adversary": 3 if default else float(10 ** np.random.uniform(-2, 1)),
            
            "autoencoder_lr": 1e-4 if default else float(10 ** np.random.uniform(-5, -3)),
            "adversary_lr": 3e-4 if default else float(10 ** np.random.uniform(-5, -3)),
            "autoencoder_wd": 1e-6 if default else float(10 ** np.random.uniform(-8, -4)),
            "adversary_wd": 1e-4 if default else float(10 ** np.random.uniform(-6, -3)),
            "adversary_steps": 3 if default else int(np.random.choice([1, 2, 3, 4, 5])),  # To not be confused: the number of adversary steps before the next VAE step 
            "batch_size": 128 if default else int(np.random.choice([64, 128, 256, 512])),
            "step_size_lr": 45 if default else int(np.random.choice([15, 25, 45])),
            "embedding_encoder_width": 512,
            "embedding_encoder_depth": 0,
            "warmup_steps": 5 if default else int(np.random.choice([0, 5, 10, 15])), 
            "data_driven_sigma": True if default else np.random.choice([True, False]),
            "ae_pretrain": True if default else np.random.choice([True, False]),
            "ae_pretrain_steps": 5
        }
        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams

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
        rec = gaussian_nll(X_hat, self.log_scale, X).sum((1,2,3)).mean()  # Single value (not averaged across batch element)
        return rec


    # Forward pass    
    def forward_ae(self, X):
        """Simple encoding-decoding process

        Args:
            X (torch.Tensor): The image data X

        Returns:
            dict: The reconstructed input, the latent representation and the losses  
        """
        if not self.variational:
            z = self.encoder(X)
        else:
            mu, log_sigma = self.encoder(X)
            # Apply reparametrization trick
            z = self.reparameterize(mu, log_sigma)
        # Decode z
        out = self.decoder(z)
        if self.variational:
            ae_loss = self.ae_loss(X, out, mu, log_sigma)
        else:
            ae_loss = self.ae_loss(X, out)
        return  dict(out=out, z=z, loss=ae_loss)

    
    def forward_compert(self, X, y_adv, drug_ids):
        """The forward step with adversarial training

        Args:
            X (torch.Tensor): the image data X
            y_adv (torch.Tensor): the target for the adversarial training  
            drug_embedding (torch.Tensor): The pre-computed embeddings for the drugs in the batch
        """
        # Autoencoder pass
        if self.variational:
            mu, log_sigma = self.encoder(X)  # Encode image
            z_basal = self.reparameterize(mu, log_sigma)  # Reparametrization trick 
        else:
            z_basal = self.encoder(X)

        # Prediction of the drug label  
        y_adv_hat = self.adversary_drugs(z_basal)

        # Embed the drug
        drug_embedding = self.drug_embeddings(drug_ids)  # Embedding weights from drug id 
        z_drug = self.drug_embedding_encoder(drug_embedding)  # Embed input drug

        # Sum the latents of the drug and the image 
        z = z_basal + z_drug  
        out = self.decoder(z) 

        # Compute the adversarial loss
        if not self.predict_n_cells:
            adv_loss = self.loss_adversary_drugs(y_adv_hat, torch.argmax(y_adv, dim = 1))
        else:
            adv_loss = self.loss_adversary_drugs(y_adv_hat, y_adv)

        # The autoencoder loss 
        if self.variational:
            ae_loss = self.ae_loss(X, out, mu, log_sigma)
        else:
            ae_loss = self.ae_loss(X, out)

        # Check whether to perform adversarial training 
        if (self.iteration % self.hparams["adversary_steps"]) != 0:
            # Compute the gradient penalty for the drug regularization term 
            adv_drugs_grad_penalty = self.compute_gradient_penalty(y_adv_hat.sum(), z_basal)
            loss = adv_loss + self.hparams["penalty_adversary"] * adv_drugs_grad_penalty
        
        else:
            loss = ae_loss['total_loss'] - self.hparams["reg_adversary"] * adv_loss
        
        return dict(out=out, z_basal=z_basal, loss=loss, ae_loss=ae_loss, adv_loss=adv_loss)

    
    def evaluate(self, X, drug_id=None):
        """Perform evaluation step

        Args:
            X (torch.tensor): The batch of observations
            y_adv (torch.tensor): The drug label of the observation 
            drug_embed (torch.tensor): The embedding of the drug 
        """
        with torch.no_grad():
            # Activate autoencoder
            if self.variational:
                mu, log_sigma = self.encoder(X)
                z_basal = self.reparameterize(mu, log_sigma)
            else:
                z_basal = self.encoder(X)

            # If not adversarial training, simply return the autoencoder losses
            if not self.adversarial:
                out = self.decoder(z_basal)
                if self.variational:
                    ae_loss = self.ae_loss(X, out, mu, log_sigma)
                else:
                    ae_loss = self.ae_loss(X, out)
                return dict(out=out, z=z_basal, ae_loss=ae_loss)

            # If adversarial training, exploit both the drug encoding and the image encoding
            else:
                y_hat = self.adversary_drugs(z_basal)
                drug_embedding = self.drug_embeddings(drug_id)
                z_drug = self.drug_embedding_encoder(drug_embedding)  # Embed input drug
                # Sum the latents of the drug and the image 
                z = z_basal + z_drug  
                # Get both the decoded versions of z and z_basal to compare them in the reconstruction 
                out = self.decoder(z)
                out_basal = self.decoder(z_basal)
                if self.variational:
                    ae_loss = self.ae_loss(X, out, mu, log_sigma)
                else:
                    ae_loss = self.ae_loss(X, out)

                return dict(out=out, out_basal=out_basal, z_basal=z_basal, z=z, y_hat=y_hat, 
                            ae_loss=ae_loss, z_drug=z_drug)
            
        
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
    
    
    def early_stopping(self, score):
        """
        Possibly early-stops training.
        """
        cond = (score > self.best_score).item()
        if cond:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1
        return cond, self.patience_trials > self.patience

    
    def update_model(self, train_loader, epoch):
        """
        Compute a forward step and returns the losses 
        """
        if self.adversarial:
            training_loss = 0  # adv + ae loss
            tot_recons_loss = 0
            tot_kl_loss = 0
            tot_ae_loss = 0
            tot_adv_loss = 0
        else:
            training_loss = 0
            if self.variational:
                tot_recons_loss = 0
                tot_kl_loss = 0
        
        # Reset the previously defined metrics
        self.metrics.reset()

        for batch in tqdm(train_loader):
            # Collect the image data from the batch 
            X = batch['X'].to(self.device)

            # If no adversarial training, simply perform the AE training loop
            if not self.adversarial:
                res = self.forward_ae(X)
                out, _, ae_loss = res.values()  # Collect the losses 
                loss = ae_loss['total_loss']
                # Optimizer step 
                self.optimizer_autoencoder.zero_grad()
                loss.backward()
                self.optimizer_autoencoder.step()

            else:
                # self.binary is true when the task is predicting trt vs dmso 
                if self.predict_n_cells:
                    y_adv = batch['n_cells'].to(self.device).float()
                else:
                    y_adv = batch['mol_one_hot'].to(self.device).long()
                
                # drug id necessary to extract embedding 
                drug_id = batch["smile_id"].to(self.device)
                del batch # Free memory
                
                # Forward pass
                out, _, loss, ae_loss, adv_loss = self.forward_compert(X, y_adv, drug_id).values()

                # Cumulate the separate adversarial and training losses 
                tot_ae_loss += ae_loss['total_loss'].item()
                tot_adv_loss += adv_loss.item()
                
                # Optimizer step - adversarial training 
                if (self.iteration % self.hparams["adversary_steps"]) != 0:       
                    self.optimizer_adversaries.zero_grad()
                    loss.backward()
                    self.optimizer_adversaries.step()
                # Optimizer step - non-adversarial training 
                else:
                    self.optimizer_autoencoder.zero_grad()
                    loss.backward()
                    self.optimizer_autoencoder.step()
            
            # Increase number of iterations
            self.iteration += 1

            # Update the global losses 
            training_loss += loss.item()
            if self.variational:
                tot_recons_loss += ae_loss['reconstruction_loss'].item()
                tot_kl_loss += ae_loss['KLD'].item()

            # Update the RMSE score of the reconstruction compared to the original 
            self.metrics.update_rmse(X, out)

        # Print the loss metrics 
        avg_loss = training_loss/len(train_loader)
        
        # Update the reconstruction loss and KL divergence according to whether we 
        # are using a variational autoencoder or a deterministic one 
        if self.variational:
            avg_kl_loss = tot_kl_loss/len(train_loader)
            avg_recon_loss = tot_recons_loss/len(train_loader)
        else:
            if not self.adversarial:
                avg_recon_loss = training_loss/len(train_loader)
            else:
                avg_recon_loss = tot_ae_loss/len(train_loader)

        self.metrics.update_bpd(avg_recon_loss)

        # The average RMSE between the evaluated images for all seen batches
        self.metrics.metrics['rmse'] /= len(train_loader)

        print(f'Mean loss after epoch {epoch}: {avg_loss}')
        if self.variational:
            print(f'Mean reconstruction loss after epoch {epoch}: {avg_recon_loss}')
            print(f'Mean kl divergence after epoch {epoch}: {avg_kl_loss}')

        # Print evaluation metrics 
        self.metrics.print_metrics()

        if self.adversarial:
            avg_ae_loss = tot_ae_loss/len(train_loader)
            avg_adv_loss = tot_adv_loss/len(train_loader)
            print(f'Mean autoencoder loss after epoch {epoch}: {avg_ae_loss}')
            print(f'Mean adversarial loss after epoch {epoch}: {avg_adv_loss}')
            if self.variational:
                return dict(loss=avg_loss, recon_loss=avg_recon_loss, kl_loss=avg_kl_loss, avg_ae_loss=avg_ae_loss, avg_adv_loss=avg_adv_loss), self.metrics.metrics
            else:
                return dict(loss=avg_loss, avg_ae_loss=avg_recon_loss, avg_adv_loss=avg_adv_loss), self.metrics.metrics
        else: 
            if self.variational:
                return dict(loss=avg_loss, recon_loss=avg_recon_loss, kl_loss=avg_kl_loss), self.metrics.metrics
            else:
                return dict(loss=avg_recon_loss), self.metrics.metrics


    def save_history(self, epoch, losses, metrics, fold):
        """Save partial model results in the history dictionary (model attribute) 

        Args:
            epoch (int): the current epoch 
            losses (dict): dictionary containing the partial losses of the model 
            metrics (dict): dictionary containing the partial metrics of the model 
            fold (str): train or valid
        """
        self.history[fold]["epoch"].append(epoch)
        # Append the losses to the right fold dictionary 
        for loss in losses:
            if loss not in self.history[fold]:
                self.history[fold][loss] = [losses[loss]]
            else:
                self.history[fold][loss].append(losses[loss])
        
        # Append the metrics to the right fold dictionary 
        for metric in metrics:
            if metric not in self.history[fold]:
                self.history[fold][metric] = [metrics[metric]]
            else:
                self.history[fold][metric].append(metrics[metric])

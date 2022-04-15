import json
from tqdm import tqdm
import numpy as np
import torch

from .modules.building_blocks import *

from .template_model import *

import sys
sys.path.insert(0, '..')

from .autoencoders import initialize_encoder_decoder
from metrics.metrics import *


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
            append_layer_width=None,
            drug_embeddings = None,
            variational: bool = True, 
            dataset_name: str = 'cellpainting',
            predict_moa: bool = False,
            n_moa: int = 0,
            total_iterations: int = None, 
            class_weights: dict = None):     

        super(CPA, self).__init__() 
        self.adversarial = False  # If a normal adversarial network must be trained (starting at false)
        self.in_width = in_width  # Image width 
        self.in_height = in_height  # Image height
        self.in_channels = in_channels  # Image channels (5 in this case)
        self.device = device

        # Parameters for the adversarial training 
        self.num_drugs = num_drugs  # num_drugs and n_seen_drugs is the same on the BBBC021 dataset 
        self.n_seen_drugs = n_seen_drugs
        self.seed = seed
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
            self.set_hparams_(0, hparams)

        # Setup metrics object
        self.metrics = TrainingMetrics(self.in_height, self.in_width, self.in_channels, self.hparams["latent_dim"], device = self.device)

        # The log sigma as an external parameter of the sigma vae training paradigm 
        if self.hparams["data_driven_sigma"]:
            self.log_scale = 0
        else:
            self.log_scale = torch.nn.Parameter(torch.full((1,1,1,1), 0.0), requires_grad=True)
        
        # True if variational inference is performed through VAE 
        self.variational = variational  

        # The information concerning dataset and the MOA based adversarial task 
        self.dataset_name = dataset_name 
        self.predict_moa = predict_moa
        self.n_moa = n_moa

        # Instantiate the convolutional encoder and decoder modules
        self.encoder, self.decoder = initialize_encoder_decoder(self.in_channels, 
                                                                self.in_width, 
                                                                self.in_height, 
                                                                self.variational, 
                                                                self.hparams)

        # Initialize warmup params
        self.warmup_steps = self.hparams["warmup_steps"]

        # Statistics history
        self.history = {"train":{'epoch':[]}, "val":{'epoch':[]}, "test":{'epoch':[]}, "ood":{'epoch':[]}}

        # Iterations to decide when to perform adversarial training 
        self.iteration = 0
        self.total_iterations = total_iterations

        # Get the parameters of a model if a condition is verified
        self.get_params = lambda model, cond: list(model.parameters()) if cond else []  

        # Class weights 
        self.class_weights = class_weights 

        # Initialize the autoencoder first with adversarial equal to False
        self.initialize_ae() 


    def initialize_ae(self):
        """Initialize autoencoder model 
        """
        print('Initalize autoencoder')
        _parameters = (
            self.get_params(self.encoder, True)
            + self.get_params(self.decoder, True)
            + self.get_params(self.log_scale, not self.hparams["data_driven_sigma"])
        )
        # Setup the optimizer and scheduler for the standard VAE 
        self.optimizer_autoencoder = torch.optim.Adam(
            _parameters,
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
        self.adversary_drugs = MLP(
            [self.hparams["latent_dim"]]
            + [self.hparams["adversary_width_drug"]] * self.hparams["adversary_depth_drug"]
            + [self.n_seen_drugs], self.hparams["batch_norm_adversarial_drug"]
        ).to(self.device)

        if self.predict_moa:
            # Adversary network for the MOA
            self.adversary_moa = MLP(
                [self.hparams["latent_dim"]]
                + [self.hparams["adversary_width_moa"]] * self.hparams["adversary_depth_moa"]
                + [self.n_moa], self.hparams["batch_norm_adversarial_moa"]
            ).to(self.device)

        # Create drug embeddings 
        if self.drug_embeddings is None:
            self.drug_embeddings = torch.nn.Embedding(
                self.num_drugs, self.hparams["drug_embedding_dimension"]).to(self.device)
            embedding_requires_grad = True
        else:
            self.drug_embeddings = self.drug_embeddings  # From pre-trained 
            embedding_requires_grad = False
        
        # Drug embedding encoder 
        self.drug_embedding_encoder = MLP(
            [self.drug_embeddings.embedding_dim]
            + [self.hparams["drug_embedding_dimension"]]
            * self.hparams["drug_embedding_encoder_depth"]
            + [self.hparams["latent_dim"]],
            last_layer_act="linear",
        ).to(self.device)

        # Embed the MOA
        if self.predict_moa:
            self.moa_embeddings = torch.nn.Embedding(
                self.n_moa, self.hparams["moa_embedding_dimension"]).to(self.device)

            # MOA embedding encoder 
            self.moa_embedding_encoder = MLP(
                [self.moa_embeddings.embedding_dim]
                + [self.hparams["moa_embedding_dimension"]]
                * self.hparams["moa_embedding_encoder_depth"]
                + [self.hparams["latent_dim"]],
                last_layer_act="linear",
            ).to(self.device)


        # Crossentropy loss for the prediction of both MOA and the drug 
        if self.hparams["weigh_loss"]:
            self.loss_adversary_drugs = torch.nn.CrossEntropyLoss(weight = torch.tensor(self.class_weights['drugs']).float().to(self.device), 
                                                                    reduction = 'mean')
            if self.predict_moa:
                self.loss_adversary_moas = torch.nn.CrossEntropyLoss(weight = torch.tensor(self.class_weights['moas']).float().to(self.device), 
                                                                    reduction = 'mean')
        else:
            self.loss_adversary_drugs = torch.nn.CrossEntropyLoss(reduction = 'mean')
            if self.predict_moa:
                self.loss_adversary_moas = torch.nn.CrossEntropyLoss(reduction = 'mean')

        
        # Collect parameters of the autoencoder branch
        _parameters = (
            self.get_params(self.encoder, True)
            + self.get_params(self.decoder, True) +
            self.get_params(self.drug_embeddings, embedding_requires_grad)
            + self.get_params(self.drug_embedding_encoder, True)
        )

        if self.predict_moa:
            _parameters.extend(self.get_params(self.moa_embeddings, True) + 
                                self.get_params(self.moa_embedding_encoder, True))

        # Optimizer for the autoencoder 
        self.optimizer_autoencoder = torch.optim.Adam(
            _parameters,
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"],
        )

        # Optimizer for the drug adversary
        _parameters = self.get_params(self.adversary_drugs, True)
        if self.predict_moa:
            _parameters.extend(self.get_params(self.adversary_moa, True))

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
        self.current_adversary_steps = self.hparams["adversary_steps"]

        # Initialize total iterations and linear annealing schedule
        if self.hparams['anneal_adv_steps']:
            self.total_iterations = self.total_iterations//2  # Reach the minimum halfway through the iterations
            self.step = (self.hparams["adversary_steps"]-self.hparams["final_adv_steps"])//self.total_iterations
            

    
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
            "autoencoder_type": 'convnet' if default else np.random.choice(['resnet', 'convnet']),
            "resnet_type": 'resnet18' if default else np.random.choice(['resnet18', 'convnet34', 'convnet50']), 
            "latent_dim": 512 if default else int(np.random.choice([256, 512, 1024])),
            "init_fm": 64 if default else int(np.random.choice([32, 64, 128])),  # Will be upsampled depth times
            "n_conv": 3 if default else int(np.random.choice([3, 4, 5])),
            "n_residual_blocks":12 if default else int(np.random.choice([6, 12, 18])),

            "adversary_width_drug": 128 if default else int(np.random.choice([64, 128, 256])),
            "adversary_depth_drug": 3 if default else int(np.random.choice([2, 3, 4])),
            "adversary_width_moa": 128 if default else int(np.random.choice([64, 128, 256])),
            "adversary_depth_moa": 3 if default else int(np.random.choice([2, 3, 4])),

            "beta": 1 if default else np.random.choice([1, 5, 10]),
            "reg_adversary": 5 if default else float(10 ** np.random.uniform(-2, 2)),  # Regularization 
            "penalty_adversary": 3 if default else float(10 ** np.random.uniform(-2, 1)),
            "dropout_ae": False if default else np.random.choice([True, False]),
            "dropout_rate_ae": 0.1 if default else np.random.choice([0.1, 0.5, 0.8]),
            "autoencoder_lr": 1e-4 if default else float(10 ** np.random.uniform(-5, -3)),
            "adversary_lr": 3e-4 if default else float(10 ** np.random.uniform(-5, -3)),
            "autoencoder_wd": 1e-6 if default else float(10 ** np.random.uniform(-8, -4)),
            "adversary_wd": 1e-4 if default else float(10 ** np.random.uniform(-6, -3)),
            "adversary_steps": 3 if default else int(np.random.choice([1, 2, 3, 4, 5])),  # To not be confused: the number of adversary steps before the next VAE step 
            "anneal_adv_steps": True if default else np.random.choice([True, False]),
            "final_adv_steps": 1 if default else np.random.choice([1, 10, 20]),
            "step_size_lr": 45 if default else int(np.random.choice([15, 25, 45])),

            "concat_embeddding": False if default else np.random.choice([False, True]),
            "concat_one_hot": False if default else np.random.choice([False, True]),
            "drug_embedding_encoder_depth": 0 if default else int(np.random.choice([0, 1, 2, 3])),
            # "drug_embedding_encoder_width": 512 if default else int(np.random.choice([128, 256, 512])),
            "moa_embedding_encoder_depth": 0 if default else int(np.random.choice([0, 1, 2, 3])),   
            # "moa_embedding_encoder_width": 512 if default else int(np.random.choice([128, 256, 512])),        
            "drug_embedding_dimension": 128 if default else int(np.random.choice([128, 256, 512])),
            "moa_embedding_dimension": 128 if default else int(np.random.choice([128, 256, 512])),

            "warmup_steps": 5 if default else int(np.random.choice([0, 5, 10, 15])), 
            "data_driven_sigma": True if default else np.random.choice([True, False]),
            "ae_pretrain": True if default else np.random.choice([True, False]),
            "ae_pretrain_steps": 5 if default else np.random.choice([1, 3, 5]),
            "batch_norm_adversarial_drug": True if default else np.random.choice([True, False]),
            "batch_norm_adversarial_moa": True if default else np.random.choice([True, False]),
            "batch_norm_layers_ae": True if default else np.random.choice([True, False]),
            "mean_recon_loss": False if default else np.random.choice([True, False]), 
            "weigh_loss": False if default else np.random.choice([True, False])
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
        # Compute prediction by the encoder 
        if not self.variational:
            z = self.encoder(X)
        else:
            z = self.encoder(X)
            mu, log_sigma = z[-1]
            # Apply reparametrization trick if VAE
            z[-1] = self.reparameterize(mu, log_sigma)

        # Decode the latent 
        out = self.decoder(z)
        if self.variational:
            ae_loss = self.ae_loss(X, out, mu, log_sigma)
        else:
            ae_loss = self.ae_loss(X, out)
        return  dict(out=out, z=z, loss=ae_loss)

    
    def forward_compert(self, X, y_adv_drug, drug_ids, y_adv_moa=None, moa_ids=None, mode='train'):
        """The forward step with adversarial training
        Args:
            X (torch.Tensor): the image data X
            y_adv (torch.Tensor): the target for the adversarial training  
            drug_embedding (torch.Tensor): The pre-computed embeddings for the drugs in the batch
            mode (str): train or eval
        """
        # Autoencoder pass
        if self.variational:
            z_basal = self.encoder(X)
            mu, log_sigma = z_basal[-1]
            # Apply reparametrization trick if VAE
            z_basal[-1] = self.reparameterize(mu, log_sigma)
        else:
            z_basal = self.encoder(X)

        # Prediction of the drug label  
        y_adv_hat_drug = self.adversary_drugs(z_basal[-1])
        # If applicable, prediction on the MOA label
        if self.predict_moa:
            y_adv_hat_moa = self.adversary_moa(z_basal[-1])

        # Embed the drug
        drug_embedding = self.drug_embeddings(drug_ids) 
        z_drug = self.drug_embedding_encoder(drug_embedding) 

        #Embed moa if applicable 
        if self.predict_moa:
            moa_embedding = self.moa_embeddings(moa_ids) 
            z_moa = self.moa_embedding_encoder(moa_embedding) 
        else:
            z_moa = 0  # If no moa in the dataset, z_moa set to null

        # Sum the latents of the drug and the image
        z = z_basal[:]
        z[-1] = z[-1] + z_drug + z_moa    
        
        # Decode z for the output 
        out = self.decoder(z) 

        # Compute the adversarial loss
        if mode == 'train':
            adv_loss_drug = self.loss_adversary_drugs(y_adv_hat_drug, y_adv_drug.argmax(1))
            adv_loss_moa = self.loss_adversary_moas(y_adv_hat_moa, y_adv_moa.argmax(1)) if self.predict_moa else 0

        # The autoencoder loss 
        if self.variational:
            ae_loss = self.ae_loss(X, out, mu, log_sigma)
        else:
            ae_loss = self.ae_loss(X, out)
        
        if mode == 'train':
        # Check whether to perform adversarial training 
            if (self.iteration % np.around(self.current_adversary_steps)) != 0:
                # Compute the gradient penalty for the drug regularization term 
                adv_drugs_grad_penalty = self.compute_gradient_penalty(y_adv_hat_drug.sum(), z_basal)
                # Compute the gradient penalty for the moa term, if applicable
                adv_moa_grad_penalty = self.compute_gradient_penalty(y_adv_hat_moa.sum(), z_basal)  if self.predict_moa else 0

                # The adversary component will be equal to 0 if predict_moa is false
                loss = adv_loss_drug + self.hparams["penalty_adversary"] * adv_drugs_grad_penalty \
                        + adv_loss_moa + self.hparams["penalty_adversary"] * adv_moa_grad_penalty

            else:
                loss = ae_loss['total_loss'] - self.hparams["reg_adversary"] * adv_loss_drug - \
                        self.hparams["reg_adversary"] * adv_loss_moa
            
            return dict(out=out, loss=loss, ae_loss=ae_loss, adv_loss_drug=adv_loss_drug, adv_loss_moa=adv_loss_moa)
        
        else:
            # If validation is performed, we also decode the basal encoding to compare it to the original 
            out_basal = self.decoder(z_basal)
            return dict(out=out, out_basal=out_basal, z_basal=z_basal[:,:self.hparams["latent_dim"]], z=z, y_hat_drug=y_adv_hat_drug, 
                        y_hat_moa=y_adv_hat_moa, ae_loss=ae_loss, z_drug=z_drug, z_moa = z_moa)
            
        
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
        # Keep incremental count of the number of times the score detected is not better than the 
        # so far best score 
        cond = (score > self.best_score)
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
        training_loss = 0  # adv + ae loss
        if self.adversarial:
            tot_ae_loss = 0 
            tot_adv_loss_drug = 0
            tot_adv_loss_moa = 0 

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
                # From the batch, collect the MOA and the drug one hots
                y_adv_drug = batch['mol_one_hot'].to(self.device).long()
                # drug id necessary to extract embedding 
                drug_id = batch["smile_id"].to(self.device)

                # MOA data are present only on one of the two datasets                 
                if self.dataset_name == 'BBBC021':
                    y_adv_moa = batch['moa_one_hot'].to(self.device).long()
                    moa_id = batch['moa_id'].to(self.device)
                else:
                    y_adv_moa = None
                    moa_id = None 
                del batch # Free memory
                
                # Forward pass
                out, loss, ae_loss, adv_loss_drug, adv_loss_moa = self.forward_compert(X, y_adv_drug, drug_id, y_adv_moa, moa_id, mode='train').values()

                # Zero-grad both optimizers (reduce memory burden)
                self.optimizer_adversaries.zero_grad()
                self.optimizer_autoencoder.zero_grad()

                # Optimizer step - adversarial training 
                if (self.iteration % np.around(self.current_adversary_steps)) != 0: 
                    loss.backward()
                    self.optimizer_adversaries.step()
                # Optimizer step - non-adversarial training 
                else:
                    loss.backward()
                    self.optimizer_autoencoder.step()

                # Cumulate the adversarial and the ae loss
                tot_ae_loss += ae_loss['total_loss'].item()
                tot_adv_loss_drug += adv_loss_drug.item()
                if self.predict_moa:
                    tot_adv_loss_moa += adv_loss_moa.item()
            
            # Update the global losses 
            training_loss += loss.item()  # total loss 
            if self.variational:
                # If the AE is not variational, total_ae_loss is already the reconstruction 
                tot_recons_loss += ae_loss['reconstruction_loss'].item()
                tot_kl_loss += ae_loss['KLD'].item()

            # Update the RMSE score of the reconstruction compared to the original 
            self.metrics.update_rmse(X, out)

            # Increase number of iterations
            self.iteration += 1

        # Average the losses 
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

        # Update the bit per dimension metric
        self.metrics.update_bpd(avg_recon_loss)  

        # The average RMSE between the evaluated images for all seen batches
        self.metrics.metrics['rmse'] /= len(train_loader)

        #Print the autoencoder metrics
        print(f'Mean loss after epoch {epoch}: {avg_loss}')
        if self.variational:
            print(f'Mean reconstruction loss after epoch {epoch}: {avg_recon_loss}')
            print(f'Mean kl divergence after epoch {epoch}: {avg_kl_loss}')

        # Print evaluation metrics 
        self.metrics.print_metrics()

        if self.adversarial:
            # Average the adversarial losses on the batch if variational
            avg_ae_loss = tot_ae_loss/len(train_loader)
            avg_adv_loss_drug = tot_adv_loss_drug/len(train_loader)
            if self.predict_moa:
                avg_adv_loss_moa = tot_adv_loss_moa/len(train_loader)       
            else:
                avg_adv_loss_moa = None 

            print(f'Mean autoencoder loss after epoch {epoch}: {avg_ae_loss}')
            print(f'Mean drug adversarial loss after epoch {epoch}: {avg_adv_loss_drug}')
            if self.predict_moa:
                print(f'Mean moa adversarial loss after epoch {epoch}: {avg_adv_loss_moa}')
            if self.variational:
                return dict(loss=avg_loss, recon_loss=avg_recon_loss, kl_loss=avg_kl_loss, avg_ae_loss=avg_ae_loss, 
                            avg_adv_loss_drug=avg_adv_loss_drug, avg_adv_loss_moa=avg_adv_loss_moa), self.metrics.metrics
            else:
                return dict(loss=avg_loss, avg_ae_loss=avg_recon_loss, avg_adv_loss_drug=avg_adv_loss_drug,
                            avg_adv_loss_moa=avg_adv_loss_moa), self.metrics.metrics

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

    


import json
from tqdm import tqdm
import numpy as np
import torch

from .modules.building_blocks import *

from .template_model import *

import sys
sys.path.insert(0, '..')

from .AE import AE
from .modules.adversarial.adversarial_nets import *
from metrics.metrics import *
from utils import *


"""
The default AE/VAE class 
"""

class CPA(TemplateModel):
    def __init__(self,
            in_width: int,
            in_height: int,
            in_channels: int,
            device: str,
            n_seen_drugs:int,
            seed: int = 0,
            patience: int = 5,
            hparams="",
            variational: bool = True, 
            dataset_name: str = 'cellpainting',
            n_moa: int = 0,
            total_iterations: int = None, 
            class_weights: dict = None,
            batch_size: dict = 256, 
            normalize=False):     

        super(CPA, self).__init__() 
        self.adversarial = False  # If a normal adversarial network must be trained (starting at false)
        self.in_width = in_width  # Image width 
        self.in_height = in_height  # Image height
        self.in_channels = in_channels  # Image channels (5 in this case)
        self.device = device

        # Parameters for the adversarial training 
        self.n_seen_drugs = n_seen_drugs
        self.seed = seed

        # Parameters for early-stopping 
        self.best_score = np.inf
        self.patience = patience
        self.patience_trials = 0
        self.batch_size = batch_size

        # set hyperparameters
        if isinstance(hparams, dict):
            self.hparams = hparams
        else:
            self.set_hparams_(0, hparams)

        # Setup metrics and losses objects
        self.metrics = TrainingMetrics(self.in_height, self.in_width, self.in_channels, batch_size, device = self.device)
        self.losses = TrainingLosses()
        
        # True if variational inference is performed through VAE 
        self.variational = variational  

        # The information concerning dataset and the MOA based adversarial task 
        self.dataset_name = dataset_name 
        self.n_moa = n_moa
        self.normalize = normalize

        # Prepare initialization of the encoder and the decoder 
        if self.hparams["decoding_style"] == 'sum':
            self.extra_fm = 0  # If we perform sum in the latent space 
        else:
            if self.hparams["concatenate_one_hot"]:
                self.extra_fm = self.n_seen_drugs  # If we concatenate the one-hot encodings of the conditions, the number of added dimension is the sum of such conditions 
            else:
                self.extra_fm = self.hparams["drug_embedding_dimension"]  # Else the number of additional feature maps is the size of the drug embeddings

        # Instantiate the convolutional encoder and decoder modules
        self.autoencoder = AE(self.in_channels, self.in_width, self.in_height, self.variational, self.hparams, self.extra_fm, self.n_seen_drugs, self.device, self.normalize)

        # Initialize warmup params
        self.warmup_steps = self.hparams["warmup_steps"]

        # Statistics history
        self.history = {"train":{'epoch':[]}, "val":{'epoch':[]}, "test":{'epoch':[]}}

        # Iterations to decide when to perform adversarial training 
        self.iteration = 0
        self.total_iterations = total_iterations

        # Get the parameters of a model if a condition is verified
        self.get_params = lambda model, cond: list(model.parameters()) if cond else []  

        # Class weights for balancing 
        self.class_weights = class_weights 
        
        # The cycle and associated steps control the alternation of GAN and AE
        self.cycle, self.max_cycle = {}, -1
        
        # The latent feature maps and the latent spatial dimension
        self.init_fm = 2**(self.hparams["n_conv"]-1)*self.hparams["init_fm"]  # Feature maps in the latent space 
        self.downsample_dim = self.in_width//(2**self.hparams["n_conv"])  # Spatial dimension in the latent 

        # Get the boolean about whether the encoders are required
        self.encoded_covariates = (self.hparams["decoding_style"] == 'sum' or  (self.hparams['decoding_style'] == 'concat' and not self.hparams['concatenate_one_hot']))  

        # Initialize the autoencoder first with adversarial equal to False
        self.initialize_ae() 

        # Initializae the iterations controlling beta annealing 
        if self.hparams['anneal_beta']:
            self.total_iterations = self.total_iterations//2  # Reach the minimum halfway through the iterations
            self.step = (self.hparams["max_beta"]-self.hparams["beta"])/self.total_iterations


    def initialize_ae(self):
        """Initialize autoencoder model 
        """
        print('Initalize autoencoder')

        # Collect parameters of the autoencoder branch
        _parameters = (self.get_params(self.autoencoder, True))

        if self.encoded_covariates:
            # Create drug embedding 
            self.drug_embeddings = torch.nn.Embedding(
                self.n_seen_drugs, self.hparams["drug_embedding_dimension"]).to(self.device)

            # Drug embedding encoder 
            if self.hparams["decoding_style"] == 'sum':
                self.drug_embedding_encoder = LabelEncoder(self.downsample_dim, 
                                                self.hparams["drug_embedding_dimension"], self.init_fm).to(self.device) 

            # Linear encoder '
            elif self.hparams['decoding_style'] == 'concat' and not self.hparams['concatenate_one_hot']:
                self.drug_embedding_encoder = LabelEncoderLinear(self.hparams["drug_embedding_dimension"], self.hparams["drug_embedding_dimension"]).to(self.device) 
            

            _parameters.extend(self.get_params(self.drug_embeddings, True) + 
                                self.get_params(self.drug_embedding_encoder, True)) 

        # Optimizer for the autoencoder 
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

        # Update the cycle with the number of batches used to train the autoencoder
        self.update_cycle('AE', self.hparams['reconstruction_steps'])


    def initialize_latent_GAN(self):
        print('Initalize latent adversarial training')
        
        # Number of 3x3 convolutions needed to reduce the spatial dimension to 1        
        depth = int(np.around(np.log2(self.downsample_dim)))  

        # Initialize the discriminator for the drugs  
        self.latent_discriminator = DiscriminatorNet(
            self.init_fm, self.hparams["latent_discr_width"], depth, self.n_seen_drugs
        ).to(self.device)

        # Crossentropy loss for the prediction of the drug 
        if self.hparams["weigh_loss"]:
            weight_tensor = torch.tensor(self.class_weights).float().to(self.device)
            self.loss_latent_GAN_disc = torch.nn.CrossEntropyLoss(weight = weight_tensor, reduction = 'mean')
            self.loss_latent_GAN_gen = LabelSmoothingLoss(smoothing=1.0, reduction='mean', weight=weight_tensor)
        else:
            self.loss_latent_GAN = torch.nn.CrossEntropyLoss(reduction = 'mean')
            self.loss_latent_GAN_gen = LabelSmoothingLoss(smoothing=1.0, reduction='mean')

        # Optimizer for the drug adversary
        _parameters = self.get_params(self.latent_discriminator, True)
        
        self.optimizer_adversaries = torch.optim.Adam(
            _parameters,
            lr=self.hparams["adversary_lr"],
            weight_decay=self.hparams["adversary_wd"],
        )

        self.scheduler_adversaries = torch.optim.lr_scheduler.StepLR(
            self.optimizer_adversaries,
            step_size=self.hparams["step_size_lr"],
            gamma=0.5,
        )

        # Turn adversarial to True
        self.adversarial = True
        
        # Update the scheduling cycle
        self.update_cycle('lat_disc', self.hparams['latent_gan_steps'])


    def initialize_recons_classifier_GAN(self):
        print('Initialize recognition GAN')
        # Discriminator on the output pixel state (True vs False)
        self.discriminator = DiscriminatorClassifier(self.in_width, 
                                                        self.in_channels, 
                                                        init_fm=64, 
                                                        num_outputs_drug=self.n_seen_drugs,
                                                        device=self.device).to(self.device)
               
        # Losses for predictor and classifier
        self.recon_predictor_loss = torch.nn.BCELoss()
        self.classifier_predictor_loss = torch.nn.CrossEntropyLoss(reduction = 'mean')
    
        # Discriminator optimizer
        _parameters = (self.get_params(self.discriminator, True))
        self.discriminator_optimizer = torch.optim.Adam(
                                            _parameters,
                                            lr=self.hparams["adversary_lr"],
                                            weight_decay=self.hparams["adversary_wd"])

        # Scheduler
        self.discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
                                                self.discriminator_optimizer,
                                                step_size=self.hparams["step_size_lr"],
                                                gamma=0.5)

        # Turn adversarial to True
        self.adversarial = True

        # Update the scheduling cycle 
        self.update_cycle('discr', self.hparams['discriminator_steps'])


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
            # General training-related
            "warmup_steps": 5 if default else int(np.random.choice([0, 5, 10, 15])), 
            "ae_pretrain": True if default else np.random.choice([True, False]),
            "ae_pretrain_steps": 5 if default else np.random.choice([1, 3, 5]),
            "mean_recon_loss": False if default else np.random.choice([True, False]), 
            "batch_norm_layers_ae": True if default else np.random.choice([True, False]),
            "dropout_ae": False if default else np.random.choice([True, False]),
            "dropout_rate_ae": 0.1 if default else np.random.choice([0.1, 0.5, 0.8]),
            "weigh_loss": False if default else np.random.choice([True, False]),
            "step_size_lr": 45 if default else int(np.random.choice([15, 25, 45])),

            # General autoencoder specifics
            "autoencoder_type": 'resnet_drit' if default else np.random.choice(['resnet_drit', 'unet']),
            "init_fm": 64 if default else int(np.random.choice([32, 64, 128])),  # Will be upsampled depth times
            "n_conv": 3 if default else int(np.random.choice([3, 4, 5])),
            "n_residual_blocks": 4 if default else int(np.random.choice([2, 46, 12])),
            "autoencoder_lr": 1e-4 if default else float(10 ** np.random.uniform(-5, -3)),
            "autoencoder_wd": 1e-6 if default else float(10 ** np.random.uniform(-8, -4)),
            "beta_reconstruction": 1 if default else float(10 ** np.random.uniform(-1, 1)), 
            "reconstruction_steps": 1 if default else np.randm.choice([1, 2, 3]),

            "decoding_style": 'sum' if default else np.random.choice(['sum', 'concat']),  # If label must be encoded or added as one-hot
            "anneal_beta": True if default else np.random.choice([True, False]),
            "beta": 0 if default else np.random.choice([1, 5, 10]),
            "max_beta": 1 if default else np.random.choice([1]),
            "drug_embedding_dimension": 128 if default else int(np.random.choice([128, 256, 512])),
            "concatenate_one_hot": True if default else np.random.choice([True, False]), 
            
            # General adversaries
            "adversary_lr": 3e-4 if default else float(10 ** np.random.uniform(-5, -3)),
            "adversary_wd": 1e-4 if default else float(10 ** np.random.uniform(-6, -3)),

            "train_latent_gan": True if default else np.random.choice([True, False]), 
            "latent_discr_width": 128 if default else int(np.random.choice([64, 128, 256])),  # Controls the feature map dimensionality of the adversay network on the latent 
            "beta_latent": 0.1 if default else float(10 ** np.random.uniform(-2, 2)),  # Regularization 
            "latent_gan_steps": 3 if default else int(np.random.choice([1, 2, 3, 4, 5])),  # To not be confused: the number of adversary steps before the next VAE step 

            # GAN hparams
            "train_discriminator_classifier": True if default else np.random.choice([True, False]), 
            "beta_discriminator": 1 if default else int(np.random.choice([1, 2, 3, 4, 5])),
            "beta_classifier": 1 if default else int(np.random.choice([1, 2, 3, 4, 5])),
            "discriminator_steps": 1,

            # L2 regularization parameters
            "l2_regularize": True if default else np.random.choice([True, False]), 
            "l2_penalty": 0.01  if default else int(np.random.choice([0.01, 0.1, 1]))
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams
            

    def update_model(self, train_loader, epoch):
        """
        Call the right model update function 
        """
        
        # Reset the previously defined metrics
        self.metrics.reset()  
        self.losses.reset()
        
        for batch in tqdm(train_loader):
            # Pick the training option
            option = self.cycle[self.iteration%(self.max_cycle+1)]

            # Collect the image data from the batch 
            X = batch['X'].to(self.device)
            # From the batch, collect the drug one hots
            y_true = batch['mol_one_hot'].to(self.device).long()  
            # drug id necessary to extract embedding 
            drug_id = y_true.argmax(dim=1).to(self.device)
            # Free memory
            del batch 

            # No adversarial training 
            if not self.adversarial:
                if self.encoded_covariates:
                    y_drug = self.encode_cov_labels(drug_id)
                else:
                    y_drug = y_true
                # Update ae
                out, _, _, _, ae_loss = self.autoencoder.forward_ae(X, y_drug=y_drug, mode='train').values()
                loss = ae_loss['total_loss']
                # Optimizer step 
                self.optimizer_autoencoder.zero_grad()
                loss.backward()
                self.optimizer_autoencoder.step()

                self.metrics.update_rmse(X, out)
                self.losses.update_losses(ae_loss)

            else:
                # Update autoencoder 
                if option == 'AE':
                    out, loss, cc_loss, latent_gan_loss_gen, discr_gan_loss_gen, classif_loss_gen, l2_loss, kl_loss = self.autoencoder_step(X, 
                                                                                            y_true, 
                                                                                            drug_id).values()

                    # Update losses 
                    loss_dict = {'cc_loss': cc_loss,
                                'latent_gan_loss_gen': latent_gan_loss_gen,
                                'discr_gan_loss_gen': discr_gan_loss_gen,
                                'classif_loss_gen': classif_loss_gen,
                                'l2_loss': l2_loss,
                                'kl_loss': kl_loss
                                }   
                    
                    loss_dict = configure_loss_dict(loss_dict)  # From utils
                    self.losses.update_losses(loss_dict)

                    # Update metrics 
                    self.metrics.update_rmse(X, out)

                # Update the latent space discriminator
                elif option == 'lat_disc':
                    latent_gan_loss = self.latent_discriminator_step(X, y_true)
                    self.losses.update_losses(latent_gan_loss)
                
                # Update the discriminator 
                elif option == 'discr':
                    discr_gan_loss = self.discriminator_step(X, y_true)
                    self.losses.update_losses(discr_gan_loss)

                self.iteration += 1
                    
        # Average the losses per batch
        self.losses.average_losses()
        self.metrics.average_metrics()

        # Print losses
        self.losses.print_losses()
        self.metrics.print_metrics()

        return self.losses.loss_dict, self.metrics.metrics
                
                
    def autoencoder_step(self, X, y_adv_drug, drug_id):
        """Step to train the autoencoder model. It maximizes the reconstruction and minimizes the GAN losses 

        Args:
            X (torch.nn.Tensor): The batch of observations 
            y_adv_drug (torch.nn.Tensor): The labels for the drugs
            drug_id (torch.nn.Tensor): The ids for the drugs (to retreieve embeddings)
        """
        # Put the autoencoder back in training mode in case it was in eval after the execution of a GAN step 
        self.autoencoder.train()

        # 1. SWAP THE LABEL RANDOMLY
        y_fake = swap_attributes(y_drug=y_adv_drug, drug_id=drug_id, device=self.device)
        drug_id_fake = y_fake.argmax(1)

        # 2. GET THE RECONSTRUCTION WITH SWAPPED LABEL 
        if self.encoded_covariates:
            # Embed the drug 
            z_drug_fake = self.encode_cov_labels(drug_id_fake)
            X_fake, _, _, z_basal, ae_loss =  self.autoencoder.forward_ae(X, y_drug=z_drug_fake, mode='train').values()
        else:
            # One hot encoding 
            X_fake, _, _, z_basal, ae_loss =  self.autoencoder.forward_ae(X, y_drug=y_fake, mode='train').values()
        
        # 3. GET THE ADVERSARIAL LATENT DISCRIMINATOR LOSS 
        if self.hparams["train_latent_gan"]:
            y_adv_hat_drug = self.latent_discriminator(z_basal)

            # Adversarial loss trying to homogenise the predictor to a uniform distribution
            adv_loss_latent_drug = self.loss_latent_GAN_gen(y_adv_hat_drug, drug_id)
        else:
            adv_loss_latent_drug = 0

        # 4. GET THE GAN LOSS ON THE COUNTERFACTED EXAMPLE
        if self.hparams["train_discriminator_classifier"]:
            # Collect the discriminator losses 
            loss_fake, loss_class = self.discriminator.generator_pass(X_hat=X_fake, 
                                                                        y_fake=drug_id_fake, 
                                                                        loss_discr=self.recon_predictor_loss, 
                                                                        loss_classif=self.classifier_predictor_loss)
        else:
            loss_fake, loss_class = 0,0


        # 5. IF REQUIRED, COMPUTE AN L2 REGULARIZATION LOSS on the content 
        if self.hparams['l2_regularize']:
            loss_l2 = self._l2_regularize(z_basal)  
        else:
            loss_l2 = 0

        # 6. IF VARIATIONAL, CAPTURE KL DIVERGENCE
        if self.variational:
            kl_loss = ae_loss['kl_loss']
        else: 
            kl_loss = 0
        
        # 7. CODE BACK TO ORIGINAL DOMAIN AND CALCULATE THE RECONSTRUCTION LOSS
        if self.encoded_covariates:
            # Embed the drug 
            z_drug_true = self.encode_cov_labels(drug_id)
            X_true, _, _, z_basal, _ =  self.autoencoder.forward_ae(X_fake, y_drug=z_drug_true, mode='train').values()
        else:
            X_true, _, _, z_basal, _ =  self.autoencoder.forward_ae(X_fake, y_drug=y_adv_drug, mode='train').values() 

        loss_cc = self.autoencoder.reconstruction_loss(X_true, X)
        
        # COMPUTE THE AUTOENCODER LOSS (minimize L1, maximize discriminator losses)
        loss = self.hparams["beta_cc"] * loss_cc + self.hparams["beta_latent"] * adv_loss_latent_drug + \
            self.hparams['beta_discriminator'] * loss_fake + self.hparams['beta_classifier'] * loss_class + \
            self.hparams['l2_penalty'] * loss_l2 + self.autoencoder.beta * kl_loss

        # Backward
        self.optimizer_autoencoder.zero_grad()
        loss.backward()
        self.optimizer_autoencoder.step()
        
        return dict(out=X_fake, loss=loss, cc_loss=loss_cc, adv_loss_drug_gen=adv_loss_latent_drug,
                    loss_recon_gan_gen=loss_fake, loss_classification_gan_drug_gen=loss_class,
                    loss_l2=loss_l2, kl_loss=kl_loss)


    def latent_discriminator_step(self, X, y_adv_drug):
        self.autoencoder.eval()
        # Just need to encode the images and predict on the latent
        if not self.variational:
            z_basal = self.autoencoder.encoder(X)
        else:
            mu, log_sigma = self.autoencoder.encoder(X)
            z_basal = self.autoencoder.reparameterize(mu, log_sigma)  

        # GET THE ADVERSARIAL LATENT DISCRIMINATOR ADVERSARIAL LOSS 
        y_adv_hat_drug = self.latent_discriminator(z_basal)
        loss = self.hparams["beta_latent"] * self.loss_latent_GAN_disc(y_adv_hat_drug, y_adv_drug.argmax(1))

        # Backward
        self.optimizer_adversaries.zero_grad()
        loss.backward()
        self.optimizer_adversaries.step()

        return dict(latent_gan_loss_discr=loss)
        

    def discriminator_step(self, X, y_adv_drug):
        self.autoencoder.eval()
        drug_id = y_adv_drug.argmax(1)

        # Only interested in the outpout of the decoder 
        if self.encoded_covariates:    
            # Embed the drug and moa
            z_drug = self.encode_cov_labels(drug_id)
            X_hat, _, _, _, _ =  self.autoencoder.forward_ae(X, y_drug=z_drug, mode='train').values()

        else:
            X_hat, _, _, _, _ =  self.autoencoder.forward_ae(X, y_drug=y_adv_drug, mode='train').values()

        loss_fake, loss_class = self.discriminator.discriminator_pass(X, 
                                                    X_hat, 
                                                    drug_id, 
                                                    self.recon_predictor_loss, 
                                                    self.classifier_predictor_loss)
        loss = self.hparams['beta_classifier'] * loss_class + self.hparams['beta_discriminator'] * loss_fake

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        return dict(classification_gan_loss_discr=loss_class, discriminator_gan_loss_discr=loss_fake)

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
    

    def update_cycle(self, name, number_of_steps):
        # The cycle controls what kind of training step is performed among discriminators and autoencoder 
        vals = [self.max_cycle+i for i in range(1, number_of_steps+1)]
        for val in vals:
            self.cycle[val] = name
        self.max_cycle = vals[-1]
    
    def encode_cov_labels(self, drug_id):
        # Embed the drug
        drug_embedding = self.drug_embeddings(drug_id) 
        z_drug = self.drug_embedding_encoder(drug_embedding) 
        return z_drug
    
    def early_stopping(self, score):
        """
        Possibly early-stops training.
        """
        # Keep incremental count of the number of times the score detected is not better than the 
        # so far best score 
        cond = (score < self.best_score)
        if cond:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1
        return cond, self.patience_trials > self.patience

    # From https://github.com/HsinYingLee/DRIT/blob/master/src/model.py
    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

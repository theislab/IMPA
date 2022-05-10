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
            batch_size: dict = 256):     

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

        # Prepare initialization of the encoder and the decoder 
        if self.hparams["decoding_style"] == 'sum':
            self.extra_fm = 0  # If we perform sum in the latent space 
        else:
            if self.hparams["concatenate_one_hot"]:
                self.extra_fm = self.n_seen_drugs  # If we concatenate the one-hot encodings of the conditions, the number of added dimension is the sum of such conditions 
            else:
                self.extra_fm = self.hparams["drug_embedding_dimension"]

        # Instantiate the convolutional encoder and decoder modules
        self.autoencoder = AE(self.in_channels, self.in_width, self.in_height, self.variational, self.hparams, self.extra_fm, self.n_seen_drugs, self.device)

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

        # Initialize the autoencoder first with adversarial equal to False
        self.initialize_ae() 

        # The cycle and associated steps control the alternation of GAN and AE
        self.cycle, self.max_cycle = {0:'AE'}, 0

        # Initializae the iterations controlling beta annealing 
        if self.hparams['anneal_beta']:
            self.total_iterations = self.total_iterations//2  # Reach the minimum halfway through the iterations
            self.step = (self.hparams["max_beta"]-self.hparams["beta"])/self.total_iterations



    def initialize_ae(self):
        """Initialize autoencoder model 
        """
        print('Initalize autoencoder')
        _parameters = (self.get_params(self.autoencoder, True))

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
        print('Initalize latent adversarial training')

        # Adversary is a convolutional net on the latent
        init_fm = 2**(self.hparams["n_conv"]-1)*self.hparams["init_fm"]  # Feature maps in the latent space 
        downsample_dim = self.in_width//(2**self.hparams["n_conv"])  # Spatial dimension in the latent 
        depth = int(np.around(np.log2(downsample_dim)))  # Number of 3x3 convolutions needed to reduce the spatial dimension to 1

        # Initialize the discriminator for the drugs and the moa 
        self.adversary_drugs = DiscriminatorNet(
            init_fm, self.hparams["adversary_width_drug"], depth, self.n_seen_drugs
        ).to(self.device)
        
        # Create drug embedding 
        self.drug_embeddings = torch.nn.Embedding(
            self.n_seen_drugs, self.hparams["drug_embedding_dimension"]).to(self.device)

        # Drug embedding encoder 
        if self.hparams["decoding_style"] == 'sum':
            self.drug_embedding_encoder = LabelEncoder(downsample_dim, 
                                            self.hparams["drug_embedding_dimension"], init_fm).to(self.device) 

        elif self.hparams['decoding_style'] == 'concat' and not self.hparams['concatenate_one_hot']:
            self.drug_embedding_encoder = LabelEncoderLinear(self.hparams["drug_embedding_dimension"], self.hparams["drug_embedding_dimension"]).to(self.device) 

        # Crossentropy loss for the prediction of both MOA and the drug 
        if self.hparams["weigh_loss"]:
            self.loss_adversary_drugs = torch.nn.CrossEntropyLoss(weight = torch.tensor(self.class_weights).float().to(self.device), 
                                                                    reduction = 'mean')
        else:
            self.loss_adversary_drugs = torch.nn.CrossEntropyLoss(reduction = 'mean')

        # Get the boolean about whether the encoders are required
        self.encoded_covariates = (self.hparams["decoding_style"] == 'sum' or  (self.hparams['decoding_style'] == 'concat' and not self.hparams['concatenate_one_hot']))  

        # Collect parameters of the autoencoder branch
        _parameters = (self.get_params(self.autoencoder, True))

        if self.encoded_covariates:
            _parameters.extend(self.get_params(self.drug_embeddings, True) + 
                                self.get_params(self.drug_embedding_encoder, True)) 

        # Optimizer for the autoencoder 
        self.optimizer_autoencoder = torch.optim.Adam(
            _parameters,
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"],
        )
        
        # Optimizer for the drug adversary
        _parameters = self.get_params(self.adversary_drugs, True)
    
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
        
        # Update the scheduling cycle
        self.update_cycle('ADV', self.hparams['adversary_steps'])


    def initialize_recons_GAN(self):
        print('Initialize recognition GAN')
        # Discriminator on the output pixel state (True vs False)
        self.recon_predictor = GANDiscriminator(self.in_width, self.in_channels, init_fm=64, device=self.device).to(self.device)
               
        # Loss for the GAN on the pixel space
        self.recon_predictor_loss = torch.nn.BCELoss()
    
        # Reconstruction GAN is to be trained 
        _parameters = (self.get_params(self.recon_predictor, True))
        self.recon_discriminator_optimizer = torch.optim.Adam(
                                            _parameters,
                                            lr=self.hparams["adversary_lr"],
                                            weight_decay=self.hparams["adversary_wd"])

        # Scheduler
        self.recon_discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
                                                self.recon_discriminator_optimizer,
                                                step_size=self.hparams["step_size_lr"],
                                                gamma=0.5)

        # Update the scheduling cycle 
        self.update_cycle('recon_GAN', self.hparams['recon_gan_steps'])


    def initialize_classific_GAN(self):
        print('Initialize reconstruction GAN')
        # Discriminator on the output pixel state (True vs False)
        self.classifier_predictor = GANClassifier(self.in_width, self.in_channels, init_fm=64, num_outputs_drug=self.n_seen_drugs).to(self.device)

        self.classifier_predictor_loss = torch.nn.CrossEntropyLoss(reduction = 'mean')

        # GAN discriminator optimizers 
        _parameters = (self.get_params(self.classifier_predictor, True))

        self.classifier_predictor_optimizer = torch.optim.Adam(
                                            _parameters,
                                            lr=self.hparams["adversary_lr"],
                                            weight_decay=self.hparams["adversary_wd"])

        # Scheduler
        self.classifier_predictor_scheduler = torch.optim.lr_scheduler.StepLR(
                                                self.classifier_predictor_optimizer,
                                                step_size=self.hparams["step_size_lr"],
                                                gamma=0.5,
                                            )
        
        # Update the scheduling cycle  
        self.update_cycle('classific_GAN', self.hparams['classification_gan_steps'])


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

            # General autoencoder specifics
            "autoencoder_type": 'resnet_drit' if default else np.random.choice(['resnet_dri', 'unet']),
            "init_fm": 64 if default else int(np.random.choice([32, 64, 128])),  # Will be upsampled depth times
            "n_conv": 3 if default else int(np.random.choice([3, 4, 5])),
            "n_residual_blocks": 4 if default else int(np.random.choice([2, 46, 12])),
            "autoencoder_lr": 1e-4 if default else float(10 ** np.random.uniform(-5, -3)),
            "autoencoder_wd": 1e-6 if default else float(10 ** np.random.uniform(-8, -4)),
            "beta_reconstruction": 1 if default else float(10 ** np.random.uniform(-1, 1)), 
            "step_size_lr": 45 if default else int(np.random.choice([15, 25, 45])),
            "decoding_style": 'sum' if default else np.random.choice(['sum', 'concat']),  # If label must be encoded or added as one-hot
            "anneal_beta": True if default else np.random.choice([True, False]),
            "beta": 0 if default else np.random.choice([1, 5, 10]),
            "max_beta": 1 if default else np.random.choice([1]),
            
            
            # Adversaries
            "adversary_width_drug": 128 if default else int(np.random.choice([64, 128, 256])),  # Controls the feature map dimensionality of the adversay network on the latent 
            "adversary_lr": 3e-4 if default else float(10 ** np.random.uniform(-5, -3)),
            "adversary_wd": 1e-4 if default else float(10 ** np.random.uniform(-6, -3)),
            "beta_latent": 0.1 if default else float(10 ** np.random.uniform(-2, 2)),  # Regularization 
            "adversary_steps": 3 if default else int(np.random.choice([1, 2, 3, 4, 5])),  # To not be confused: the number of adversary steps before the next VAE step 
            "drug_embedding_dimension": 128 if default else int(np.random.choice([128, 256, 512])),
            "concatenate_one_hot": True if default else np.random.choice([True, False]),            
            "recon_gan": True if default else np.random.choice([True, False]),
            "classification_gan": True if default else np.random.choice([True, False]),
            "beta_gan_recon": 1,
            "beta_gan_classific": 1,
            "recon_gan_steps": 1,
            "classification_gan_steps":1
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams
            
    
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

            # No adversarial training 
            if not self.adversarial:
                del batch
                # Update ae
                out, _, _, _, ae_loss = self.autoencoder.forward_ae(X, y_drug=None, mode='train').values()
                loss = ae_loss['total_loss']
                # Optimizer step 
                self.optimizer_autoencoder.zero_grad()
                loss.backward()
                self.optimizer_autoencoder.step()

                self.metrics.update_rmse(X, out)
                self.losses.update_losses(ae_loss)

            else:
                # From the batch, collect the drug one hots
                y_adv_drug = batch['mol_one_hot'].to(self.device).long()  
                # drug id necessary to extract embedding 
                drug_id = y_adv_drug.argmax(dim=1).to(self.device)

                del batch # Free memory

                # Update autoencoder 
                if option == 'AE':
                    out, loss, ae_loss, adv_loss_drug, loss_recon_gan, loss_classification_gan= self.autoencoder_step(X, 
                                                                                            y_adv_drug, 
                                                                                            drug_id).values()

                    # Update losses 
                    loss_dict = {'ae_loss': ae_loss,
                                'adv_loss_drug': adv_loss_drug,
                                'loss_recon_gan': loss_recon_gan,
                                'loss_classification_gan': loss_classification_gan
                                }   
                    
                    loss_dict = configure_loss_dict(loss_dict)  # From utils
                    self.losses.update_losses(loss_dict)

                    # Update metrics 
                    self.metrics.update_rmse(X, out)

                    # Update the bit per dimension metric
                    recon_loss = loss_dict['recon_loss'] if self.variational else loss_dict['total_loss']

                # Update the latent space discriminator
                elif option == 'ADV':
                    self.latent_discriminator_step(X, y_adv_drug)
                
                # Update the patch discriminator
                elif option == 'recon_GAN':
                    self.recon_GAN_step(X, y_adv_drug)

                # Update the GAN classifier
                elif option == 'classific_GAN':
                    self.classific_GAN_step(X, y_adv_drug)

                self.iteration += 1
        
        # Average the losses per batch
        self.losses.average_losses(len(train_loader))
        self.metrics.average_metrics(len(train_loader))

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

        # GET THE RECONSTRUCTION LOSS
        if self.encoded_covariates:
            # Embed the drug 
            z_drug = self.encode_cov_labels(drug_id)
            out, _, z, z_basal, ae_loss =  self.autoencoder.forward_ae(X, y_drug=z_drug, mode='train').values()

        else:
            out, _, z, z_basal, ae_loss =  self.autoencoder.forward_ae(X, y_drug=y_adv_drug, mode='train').values()

        
        # GET THE ADVERSARIAL LATENT DISCRIMINATOR LOSS 
        y_adv_hat_drug = self.adversary_drugs(z_basal)    
        adv_loss_latent_drug = self.loss_adversary_drugs(y_adv_hat_drug, drug_id)

        # GET THE ADVERSARIAL LOSS DUE TO PATCH GAN 
        if self.hparams["recon_gan"]:
            loss_recon_gan = self.recon_predictor.generator_pass(out, self.recon_predictor_loss)
        else:
            loss_recon_gan = 0 

        # GET THE CLASSIFICATION GAN LOSS 
        if self.hparams["classification_gan"]:
            # Permute labels 
            
            y_adv_drug_flip = self.swap_attributes(y_drug=y_adv_drug, drug_id = drug_id)
            drug_id_flip = y_adv_drug_flip.argmax(1)
            
            # Encode labels
            if self.encoded_covariates:
                z_drug_flip = self.encode_cov_labels(drug_id_flip)
            else:
                z_drug_flip = y_adv_drug_flip

            # Perform decoding with randomly flipped attributes
            x_flipped, _ = self.autoencoder.decoder(z_basal, z_drug_flip)

            loss_classification_gan = self.classifier_predictor.generator_pass(x_flipped, self.classifier_predictor_loss, drug_id_flip)  
        else:
            loss_classification_gan = 0
        
        # COMPUTE THE AUTOENCODER LOSS (minimize L1, maximize discriminator losses)
        loss = self.hparams["beta_reconstruction"] * ae_loss['total_loss'] - self.hparams["beta_latent"] * adv_loss_latent_drug + \
            self.hparams['beta_gan_recon']*loss_recon_gan + self.hparams['beta_gan_classific']*loss_classification_gan

        # Backward
        self.optimizer_autoencoder.zero_grad()
        loss.backward()
        self.optimizer_autoencoder.step()
        
        return dict(out=out, loss=loss, ae_loss=ae_loss, adv_loss_drug=adv_loss_latent_drug,
                    loss_recon_gan=loss_recon_gan, loss_classification_gan_drug=loss_classification_gan)


    def latent_discriminator_step(self, X, y_adv_drug):
        self.autoencoder.eval()
        # Just need to encode the images and predict on the latent
        if not self.variational:
            z_basal = self.autoencoder.encoder(X)
        else:
            mu, log_sigma = self.autoencoder.encoder(X)
            z_basal = self.autoencoder.reparameterize(mu, log_sigma)  

        # GET THE ADVERSARIAL LATENT DISCRIMINATOR ADVERSARIAL LOSS 
        y_adv_hat_drug = self.adversary_drugs(z_basal)
        loss = self.loss_adversary_drugs(y_adv_hat_drug, y_adv_drug.argmax(1))

        # Backward
        self.optimizer_adversaries.zero_grad()
        loss.backward()
        self.optimizer_adversaries.step()


    def recon_GAN_step(self, X, y_adv_drug):
        self.autoencoder.eval()

        # Only interested in the outpout of the decoder 
        if self.encoded_covariates:
            drug_id = y_adv_drug.argmax(1)
            # Embed the drug and moa
            z_drug = self.encode_cov_labels(drug_id)

            out, _, _, _, _ =  self.autoencoder.forward_ae(X, y_drug=z_drug, mode='train').values()
        else:
            out, _, _, _, _ =  self.autoencoder.forward_ae(X, y_drug=y_adv_drug, mode='train').values()

        loss = self.recon_predictor.discriminator_pass(X, out, self.recon_predictor_loss)

        self.recon_discriminator_optimizer.zero_grad()
        loss.backward()
        self.recon_discriminator_optimizer.step()


    def classific_GAN_step(self, X, y_adv_drug):
        loss =  self.classifier_predictor.discriminator_pass(X, self.classifier_predictor_loss, y_adv_drug.argmax(1))

        self.classifier_predictor_optimizer.zero_grad()
        loss.backward()
        self.classifier_predictor_optimizer.step()


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
    

    def swap_attributes(self, y_drug, drug_id):
        # Initialize the swapped indices 
        swapped_idx = torch.zeros_like(y_drug)
        # Maximum drug index
        max_drug = y_drug.shape[1] 
        # Ranges of possible drugs 
        offsets = torch.randint(1, max_drug, (drug_id.shape[0], 1)).to(self.device)
        # Permute
        permutation = drug_id + offsets
        # Remainder 
        permutation = torch.remainder(permutation, max_drug)
        # Add ones 
        swapped_idx[np.arange(y_drug.shape[0]), permutation.squeeze()] = 1
        return swapped_idx

    
    def encode_cov_labels(self, drug_id):
        # Embed the drug
        drug_embedding = self.drug_embeddings(drug_id) 
        z_drug = self.drug_embedding_encoder(drug_embedding) 
        return z_drug

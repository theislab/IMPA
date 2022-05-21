from locale import normalize
import matplotlib.pyplot as plt
import os
import torch
import sys
sys.path.insert(0, '.')

# Available autoencoder model attached to CPA 
from model.CPA import CPA
from data.fluorescent import BBBC021Dataset
from utils import *
from plot_utils import Plotter 
from evaluate import *

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np


import seml
from sacred import Experiment

# Initialize seml experiment
ex = Experiment()
seml.setup_logger(ex)

# Setup the statistics collection post experiment 
@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

# Configure the seml experiment 
@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite))

class Trainer:
    """
    Experiment wrapper around the Sacred experiment class
    """
    def __init__(self, init_all=True):
        if init_all:
            self.init_all()
    
    @ex.capture(prefix="paths")
    def init_folders(self, experiment_name, image_path, data_index_path, result_path, dataset_name):
        """Initialize the logging folders 
        Args:
            experiment_name (str): Name of the experiment 
            image_path (str): Path to the image dataset
            data_index_path (str): Path to the image metadata
            result_path (str): path to the outcome
            dataset_name (str): the name of the dataset. Can be cellpainting or BBBC021
        """
        self.experiment_name = experiment_name  
        self.image_path = image_path 
        self.data_index_path = data_index_path  
        self.result_path = result_path
        self.dataset_name = dataset_name


    @ex.capture(prefix="resume")
    def init_resume(self, resume, resume_checkpoint):
        """Initialize the resuming of an initialized run
        Args:
            resume (bool): Whether to resume a previously performed run 
            resume_checkpoint (str): The checkpoint path  
            resume_epoch (str): The epoch for the resuming 
        """
        # Resume the training 
        self.resume = resume 
        self.resume_checkpoint = resume_checkpoint  
        self.resume_epoch = 1  


    @ex.capture(prefix="training_params")
    def init_training_params(self, img_plot, save_results, num_epochs, batch_size, eval, eval_every, 
                            n_workers_loader, generate, model_name, temperature, augment_train, normalize, patience, seed):
        """Initialization of parameters for training
        Args:
            img_plot (bool): Used when trained on notebooks - print generations/reconstructions after epoch
            save_results (bool): If the images have to be saved in the result folder
            num_epochs (int): Number of epochs for training
            batch_size (int): Size of the batches
            eval (bool): Whether evaluation should occur
            eval_every (int): How often validation takes place 
            n_workers_loader (int): Number of workers for batch loading
            generate (bool): Whether to perform a sampling + decoding experiment during evaluation (only variational)
            model_name (str): Name of autoencoder model to run (AE or VAE)
            temperature (float): Temperature to downscale the random samples from the prior
            augment_train (bool): Whether augmentation should be carried out on the training set
            patience (int): How many steps of non-improvement of valid loss before stopping 
            seed (int): The random seed for reproducibility 
        """
        self.img_plot = img_plot    
        self.save_results = save_results  

        self.num_epochs = num_epochs  
        self.batch_size = batch_size
        self.eval = eval  
        self.eval_every = eval_every   

        self.n_workers_loader = n_workers_loader  
        self.generate = generate  
        self.model_name = model_name   
        self.temperature = temperature   
        self.augment_train = augment_train   
        self.normalize = normalize 

        self.patience = patience
        self.seed = seed 
        
        # Set device
        self.device = self.set_device() 
        print(f'Working on device: {self.device}')


    @ex.capture(prefix="image_params")
    def init_img_params(self, in_width, in_height, in_channels):
        """Initialize image parameters 
        Args:
            in_width (int): Spatial width
            in_height (int): Spatial height 
            in_channels (int): Number of input channels  
        """
        # Image features
        self.in_width = in_width
        self.in_height = in_height
        self.in_channels = in_channels


    def init_dataset(self):
        """Initialize dataset and data loaders
        """
        # # Prepare the data
        print('Lodading the data...') 
        self.training_set, self.validation_set, self.test_set = self.create_torch_datasets()
        
        # Create data loaders 
        self.loader_train = torch.utils.data.DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True, 
                                                    num_workers=self.n_workers_loader, drop_last=True)  # Drop last batch for a better estimate of the accuracy 
        # For validation, it is better to keep the batch size as small as possible  
        self.loader_val = torch.utils.data.DataLoader(self.validation_set, batch_size=1, shuffle=True, 
                                                    num_workers=self.n_workers_loader, drop_last=False)
        self.loader_test = torch.utils.data.DataLoader(self.test_set, batch_size=1, shuffle=True, 
                                                    num_workers=self.n_workers_loader, drop_last=False)

        # The id of the dmso to exclude from the prediction 
        self.class_imbalance_weights = self.training_set.class_imbalances

        # Mapping drug to moa
        self.drug2moa = self.training_set.couples_drug_moa 
        self.drugs2idx = self.training_set.drugs2idx
        self.idx2drugs = {value:key for key, value in self.drugs2idx.items()}
        print('Successfully loaded the data')

    
    @ex.capture(prefix="model")
    def init_model(self, hparams):      
        """Initialize the model 
        """
        self.hparams = hparams  # Dictionary with model hyperparameters 
        # Initialize model 
        self.model =  self.load_model().to(self.device)
        self.model = torch.nn.DataParallel(self.model)  # In case multiple GPUs are present
    

    def init_log(self):
        """Initialize the tensorboard loader and directories 
        """
        # Create result folder
        if self.save_results:
            if self.resume:
                self.resume_epoch = self.model.module.load_checkpoints(self.resume_checkpoint)
            print('Create output directories for the experiment')
            self.dest_dir = make_dirs(self.result_path, self.experiment_name)
        else:
            self.dest_dir = ''


    @ex.capture
    def init_all(self, seed: int):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.seed = seed
        self.init_folders()
        self.init_resume()
        self.init_training_params()
        self.init_img_params()
        self.init_model()
        self.init_log()
        

    def create_torch_datasets(self):
        """
        Create dataset compatible with the pytorch training loop 
        """
        dataset = BBBC021Dataset(self.image_path, self.data_index_path, device=self.device, 
                                                return_labels=True, augment_train=self.augment_train, normalize=self.normalize) 
        self.dim = 3  # Channel dimension

        # Number of drugs and number of modes of action 
        self.n_seen_drugs = dataset.num_drugs
        self.num_moa = dataset.num_moa

        # Collect training, test and validation sets
        training_set, validation_set, test_set = dataset.fold_datasets.values()  

        # Free cell painting dataset memory
        del dataset
        return training_set, validation_set, test_set


    @ex.capture(prefix="training")
    def train(self):
        """
        Full training loop
        """
        print('Training with hparams:')
        print(self.hparams)

        if self.save_results:
            # Setup plotting object
            self.plotter = Plotter(self.dest_dir)
        
        # The variable `end` determines whether we are at the end of the training loop 
        end = False

        print(f'Beginning training with epochs {self.num_epochs}')

        # Resume from epoch of the loaded checkpoint (1 if start from first epoch)
        for epoch in range(self.resume_epoch, self.resume_epoch+self.num_epochs+1):

            # If we are at the end of the autoencoder pretraining steps, we initialize the adversarial nets 
            if epoch > self.model.module.hparams["ae_pretrain_steps"] and not self.model.module.adversarial:
                # Initialize the latent discriminator 
                if self.hparams['train_latent_gan']:
                    self.model.module.initialize_latent_GAN()
                # Patch GAN for reconstruction accuracy 
                if self.hparams['train_discriminator_classifier']:
                    self.model.module.initialize_recons_classifier_GAN()
            
            print(f'Running epoch {epoch}')
            self.model.train() 
            
            # Losses and metrics dictionaries from the epoch 
            train_losses, train_metrics = self.model.module.update_model(self.loader_train, epoch)  # Update run
            
            # Save results to model's history  
            self.model.module.save_history(epoch, train_losses, train_metrics, 'train')

            # Evaluate
            if epoch % self.eval_every == 0 and self.eval:
                # Switch the model to evaluate mode 
                self.model.eval()

                # Get the validation results 
                val_losses, val_metrics = training_evaluation(self.model.module, self.loader_val, self.model.module.adversarial,
                                                        self.model.module.metrics, self.model.module.losses, self.device, end, 
                                                        variational=self.model.module.variational, ds_name=self.dataset_name, drug2moa=self.drug2moa)

                self.model.module.save_history(epoch, val_losses, val_metrics, 'val')

                # Plot reconstruction of a random image 
                if self.save_results and self.generate:
                    with torch.no_grad():
                        if self.hparams['decoding_style'] == 'sum' or (self.hparams['decoding_style'] == 'concat' and not self.hparams['concatenate_one_hot']):
                            original, reconstructed, reconstructed_flip, label_orig, label_swap = self.model.module.autoencoder.generate(self.loader_val,
                                                                                    self.model.module.drug_embeddings,
                                                                                    self.model.module.drug_embedding_encoder,
                                                                                    self.model.module.adversarial)  
                        else:
                            original, reconstructed, reconstructed_flip, label_orig, label_swap = self.model.module.autoencoder.generate(self.loader_val,
                                                                                    None,
                                                                                    None,
                                                                                    self.model.module.adversarial)
                    
                    self.plotter.plot_reconstruction(tensor_to_image(original), 
                                                    tensor_to_image(reconstructed), 
                                                    epoch, 
                                                    self.save_results, 
                                                    self.img_plot, 
                                                    dim=self.dim, 
                                                    size = 4,
                                                    drug1 = self.idx2drugs[label_orig.item()],
                                                    drug2 = self.idx2drugs[label_orig.item()])

                    self.plotter.plot_reconstruction(tensor_to_image(original), 
                                                    tensor_to_image(reconstructed_flip), 
                                                    str(epoch)+'_swap', 
                                                    self.save_results, 
                                                    self.img_plot, 
                                                    dim=self.dim, 
                                                    size = 4,
                                                    drug1 = self.idx2drugs[label_orig.item()],
                                                    drug2 = self.idx2drugs[label_swap.item()])
                    # plt.show()
                    del original
                    del reconstructed

                score = val_metrics['rmse']

                # Save the model if it is the best performing one 
                print(f'New best score is {self.model.module.best_score}')
                # Save the model with lowest validation loss 
                self.model.module.save_checkpoints(epoch, 
                                                    val_metrics, 
                                                    train_losses, 
                                                    val_losses, 
                                                    self.dest_dir)

                print(f"Save new checkpoint at {os.path.join(self.dest_dir, 'checkpoint')}")
            
            # Scheduler step at the end of the epoch for the autoencoder 
            if epoch <= self.hparams["warmup_steps"]:
                self.model.module.optimizer_autoencoder.param_groups[0]["lr"] = self.hparams["autoencoder_lr"] * min(1., epoch/self.model.module.warmup_steps)
            else:
                self.model.module.scheduler_autoencoder.step()

            # We do not warmup the adversaries and perform the lr scheduling for them separately 
            if self.model.module.adversarial:
                if self.hparams['train_latent_gan']:
                    self.model.module.scheduler_adversaries.step()
                if self.hparams['train_discriminator_classifier']:
                    self.model.module.discriminator_scheduler.step()

            # Update the number of adversarial steps performed per each autoencoder step 
            if self.model.module.hparams['anneal_beta']:
                # Update the number of adversarial steps so they do not go under a minumum 
                self.model.module.autoencoder.beta = min(self.model.module.hparams["max_beta"], self.model.module.autoencoder.beta+self.model.module.step)        

        # Perform last evaluation on TEST SET   
        self.model.eval()
        end = True  # Flag used to indicate the end of training, where latent results are saved

        test_losses, test_metrics = training_evaluation(self.model.module, self.loader_test, self.model.module.adversarial,
                                                self.model.module.metrics, self.model.module.losses, self.device, end, 
                                                variational=self.model.module.variational, ds_name=self.dataset_name, drug2moa=self.drug2moa, save_path=self.dest_dir)

        self.model.module.save_history('final_test', test_losses, test_metrics, 'test')

        # Get results in a correct format
        results = self.format_seml_results(self.model.module.history)
        return results
        

    def format_seml_results(self, history):
        """Format results for seml 

        Args:
            history (_dict_): dictionary containing the history of the model's statisistics
        """
        results = {}
        for fold in history:
            for stat in history[fold]:
                key = f'{fold}_{stat}'
                results[key] = history[fold][stat]
        return results

        
    def set_device(self):
        """
        Set cpu or gpu device for data and computations 
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return device

    
    def load_model(self):
        """
        Load the model from the dedicated library
        """
        # Dictionary of models 
        variational =  False if self.model_name == 'AE' else True
        return CPA(in_width = self.in_width,
                    in_height = self.in_height,
                    in_channels = self.in_channels,
                    device = self.device,
                    n_seen_drugs = self.n_seen_drugs,
                    seed = self.seed,
                    patience = self.patience,
                    hparams = self.hparams,
                    variational = variational,
                    dataset_name = self.dataset_name,
                    n_moa = self.num_moa, 
                    total_iterations = self.num_epochs,
                    class_weights = self.class_imbalance_weights,
                    batch_size = self.batch_size,
                    normalize = self.normalize)


# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print("get_experiment")
    experiment = Trainer(init_all=init_all)
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = Trainer()
    return experiment.train()

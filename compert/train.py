import os
import torch
import sys
sys.path.insert(0, '.')

# Available autoencoder model attached to CPA 
from model.sigma_VAE import SigmaVAE
from model.sigma_AE import SigmaAE
from data.cellpainting import CellPaintingDataset
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
    def init_folders(self, experiment_name, image_path, data_index_path, embeddings_path, use_embeddings, result_path, dataset_name):
        """Initialize the logging folders 
        Args:
            experiment_name (str): Name of the experiment 
            image_path (str): Path to the image dataset
            data_index_path (str): Path to the image metadata
            embeddings_path (str): The path to the molecular embedding csv
            use_embeddings (str): Whether to use pre-trained embeddings or not
            result_path (str): path to the outcome
            dataset_name (str): the name of the dataset. Can be cellpainting or BBBC021
        """
        self.experiment_name = experiment_name  
        self.image_path = image_path 
        self.data_index_path = data_index_path  
        self.embeddings_path = embeddings_path  
        self.use_embeddings = use_embeddings  
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
                            n_workers_loader, generate, model_name, temperature, augment_train, patience, seed,
                            append_layer_width=False, predict_moa = False):
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
            append_layer_width (bool, optional): Controls The addition of trailing layers to the adversarial and drug embedding MLPs. Defaults to False.
            predict_moa (book, optional): Used only if the the dataset employed is MOA-annotated 
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

        self.patience = patience
        self.seed = seed 
        
        self.append_layer_width = append_layer_width
        self.predict_moa = predict_moa
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
        self.training_set, self.validation_set, self.test_set, self.ood_set = self.create_torch_datasets()
        
        # Create data loaders 
        self.loader_train = torch.utils.data.DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True, 
                                                    num_workers=self.n_workers_loader, drop_last=True)  # Drop last batch for a better estimate of the accuracy 
        # For validation, it is better to keep the batch size as small as possible  
        self.loader_val = torch.utils.data.DataLoader(self.validation_set, batch_size=1, shuffle=True, 
                                                    num_workers=self.n_workers_loader, drop_last=False)
        self.loader_test = torch.utils.data.DataLoader(self.test_set, batch_size=1, shuffle=True, 
                                                    num_workers=self.n_workers_loader, drop_last=False)
        self.loader_ood = torch.utils.data.DataLoader(self.ood_set, batch_size=1, shuffle=True, 
                                                    num_workers=self.n_workers_loader, drop_last=False)

        # The id of the dmso to exclude from the prediction 
        self.dmso_id = self.training_set.drugs2idx['DMSO']
        self.class_imbalance_weights = self.training_set.class_imbalances 
        print('Successfully loaded the data')

    
    @ex.capture(prefix="model")
    def init_model(self, hparams):      
        """Initialize the model 
        """
        self.hparams = hparams  # Dictionary with model hyperparameters 
        # Initialize model 
        self.model =  self.load_model().to(self.device)
        print(self.model)
        self.model = torch.nn.DataParallel(self.model)  # In case multiple GPUs are present
    

    def init_log(self):
        """Initialize the tensorboard loader and directories 
        """
        # Create result folder
        if self.save_results:
            if self.resume:
                print('Create output directories for the experiment')
                self.resume_epoch = self.model.module.load_checkpoints(self.resume_checkpoint)
            self.dest_dir = make_dirs(self.result_path, self.experiment_name)
            # Setup logger in any case
            self.writer = SummaryWriter(os.path.join(self.result_path, 'logs'))

    
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
        self.init_dataset()
        self.init_model()
        self.init_log()
        

    def create_torch_datasets(self):
        """
        Create dataset compatible with the pytorch training loop 
        """
        # Create dataset objects for the three data folds
        if self.dataset_name == 'cellpainting':
            dataset = CellPaintingDataset(self.image_path, self.data_index_path, self.embeddings_path, device=self.device, 
                                                    return_labels=True, use_pretrained=self.use_embeddings, augment_train=self.augment_train)
            self.dim = 5  # The number of channels
        else:
            dataset = BBBC021Dataset(self.image_path, self.data_index_path, self.embeddings_path, device=self.device, 
                                                    return_labels=True, use_pretrained=self.use_embeddings, augment_train=self.augment_train) 
            self.dim = 3

        # Extract matrix of embeddings only if the embedding option is chosen 
        if self.use_embeddings:
            self.drug_embeddings = dataset.drug_embeddings
        else:
            self.drug_embeddings = None  # No pre-trained embeddings are used and drug embeddings are learnt

        # Collect the number of total drugs and the one of seen (the two correspond in the fluorescent microscopy dataset)
        self.num_drugs = dataset.num_drugs
        # The ood set in the BBC021 dataset does not leave out entire compounds but only compound dosage combinations 
        if self.dataset_name == 'BBBC021':
            self.n_seen_drugs = self.num_drugs
            self.num_moa = dataset.num_moa
        else:
            self.n_seen_drugs = dataset.n_seen_drugs  
            self.num_moa = 0  # No MOA annotated 
        
        # Collect training, test and validation sets
        training_set, validation_set, test_set, ood_set = dataset.fold_datasets.values()  
        # Free cell painting dataset memory
        del dataset
        return training_set, validation_set, test_set, ood_set


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
        
        # The variable `end` determines whether we are at the end of the training loop and therefore if disentanglement and clustering stats
        # are to be evaluated on the test and ood splits
        end = False

        print(f'Beginning training with epochs {self.num_epochs}')

        for epoch in range(self.resume_epoch, self.resume_epoch+self.num_epochs+1):

            # If we are at the end of the autoencoder pretraining steps, we initialize the adversarial net  
            if epoch >= self.model.module.hparams["ae_pretrain_steps"] + 1 and not self.model.module.adversarial:
                self.model.module.initialize_adversarial()
            
            print(f'Running epoch {epoch}')
            self.model.train() 
            
            # Losses and metrics dictionaries from the epoch 
            train_losses, train_metrics = self.model.module.update_model(self.loader_train, epoch)  # Update run 
            
            # Save results to tensorboard and to model's history  
            if self.save_results:
                self.write_results(train_losses, train_metrics, self.writer, epoch, 'train')
            self.model.module.save_history(epoch, train_losses, train_metrics, 'train')

            # Evaluate
            if epoch % self.eval_every == 0 and self.eval:
                # Switch the model to evaluate mode 
                self.model.eval()

                # Get the validation results 
                val_losses, val_metrics = training_evaluation(self.model.module, self.loader_val, self.model.module.adversarial,
                                                        self.model.module.metrics, self.dmso_id, self.device, end, 
                                                        variational=self.model.module.variational, predict_moa=self.predict_moa,
                                                        ds_name=self.dataset_name)

                if self.save_results:
                    self.write_results(val_losses, val_metrics, self.writer, epoch, 'val')
                self.model.module.save_history(epoch, val_losses, val_metrics, 'val')

                # Plot reconstruction of a random image 
                if self.save_results:
                    with torch.no_grad():
                        original, reconstructed = self.model.module.generate(self.loader_val)
                    self.plotter.plot_reconstruction(tensor_to_image(original), 
                                                    tensor_to_image(reconstructed), epoch, self.save_results, self.img_plot, dim=self.dim)
                    del original
                    del reconstructed
                    
                    # Plot generation of sampled images (only fir variational autoencoders) 
                    if self.generate and self.model.module.variational:
                        sampled_img = tensor_to_image(self.model.module.sample(1, self.temperature))
                        self.plotter.plot_channel_panel(sampled_img, epoch, self.save_results, self.img_plot, dim=self.dim)
                        del sampled_img

                # Decide on early stopping based on the bit/dim of the image during autoencoder mode and the difference between decoded images after
                score = val_metrics['bpd']
                
                # Evaluate early-stopping 
                cond, early_stopping = self.model.module.early_stopping(score) 

                # Save the model if it is the best performing one 
                if cond and self.save_results:
                    print(f'New best score is {self.model.module.best_score}')
                    # Save the model with lowest validation loss 
                    self.model.module.save_checkpoints(epoch, 
                                                        val_metrics, 
                                                        train_losses, 
                                                        val_losses, 
                                                        self.dest_dir)

                    print(f"Save new checkpoint at {os.path.join(self.dest_dir, 'checkpoint')}")

            # If we overcome the patience, we break the loop
            if early_stopping:
                break 
            
            # Scheduler step at the end of the epoch 
            if epoch <= self.hparams["warmup_steps"]:
                self.model.module.optimizer_autoencoder.param_groups[0]["lr"] = self.hparams["autoencoder_lr"] * min(1., epoch/self.model.module.warmup_steps)
            else:
                self.model.module.scheduler_autoencoder.step()
            # We do not warmup the adversaries
            if self.model.module.adversarial:
                self.model.module.scheduler_adversaries.step()
            
            # Update the number of adversarial steps performed per each autoencoder step 
            if self.model.module.adversarial and self.model.module.hparams['anneal_adv_steps']:
                # Update the number of adversarial steps so they do not go under a minumum 
                self.model.module.current_adversary_steps = max(self.model.module.hparams["final_adv_steps"], self.model.module.current_adversary_steps-self.model.module.step)
                self.model.module.iterations = 0 
        

        self.model.eval()
        # Perform last evaluation on TEST SET   
        end = True
        test_losses, test_metrics = training_evaluation(self.model.module, self.loader_test, self.model.module.adversarial,
                                                self.model.module.metrics, self.dmso_id, self.device, end, 
                                                variational=self.model.module.variational, ood=False, predict_moa=self.predict_moa,
                                                ds_name=self.dataset_name)
        if self.save_results:
            self.write_results(test_losses, test_metrics, self.writer, epoch, ' test')
        self.model.module.save_history('final_test', test_losses, test_metrics, 'test')

        # Perform last evaluation on OOD SET
        ood_losses, ood_metrics = training_evaluation(self.model.module, self.loader_ood, adversarial=self.model.module.adversarial,
                                                metrics=self.model.module.metrics, dmso_id=self.dmso_id, device=self.device, end=end, 
                                                variational=self.model.module.variational, ood=True, predict_moa=self.predict_moa,
                                                ds_name=self.dataset_name)
        if self.save_results:
            self.write_results(ood_losses, ood_metrics, self.writer, epoch,' ood')
        self.model.module.save_history('final_ood', ood_losses, ood_metrics, 'ood')

        # Get results in a correct format
        results = self.format_seml_results(self.model.module.history)
        return results


    def write_results(self, losses, metrics, writer, epoch, fold='train'):
        """Write results to tensorboard
        Args:
            losses (dict): Dictiornay with the loss metrics  
            metrics (dict): Dictionary with the evaluation metrics
            writer (torch.utils.tensorboard.SummaryWriter): Summary statistics writer 
        """
        for key in losses:
            writer.add_scalar(tag=f'{self.experiment_name}/{fold}/{key}', scalar_value=losses[key], 
                                    global_step=epoch)
        for key in metrics:
            writer.add_scalar(tag=f'{self.experiment_name}/{fold}/{key}', scalar_value=metrics[key], global_step=epoch)
    

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
        models = {'VAE': SigmaVAE, 'AE': SigmaAE}
        model = models[self.model_name]
        return model(in_width = self.in_width,
                    in_height = self.in_height,
                    in_channels = self.in_channels,
                    device = self.device,
                    num_drugs = self.num_drugs,
                    n_seen_drugs = self.n_seen_drugs,
                    seed = self.seed,
                    patience = self.patience,
                    hparams = self.hparams,
                    append_layer_width = self.append_layer_width,
                    drug_embeddings = self.drug_embeddings,
                    dataset_name = self.dataset_name,
                    predict_moa = self.predict_moa,
                    n_moa = self.num_moa, 
                    total_iterations = self.num_epochs,
                    class_weights = self.class_imbalance_weights)


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

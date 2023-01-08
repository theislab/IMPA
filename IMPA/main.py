import seml
import torch
from sacred import SETTINGS, Experiment
from .solver import Solver
from torch.backends import cudnn

# Avoid lists in an input configuration to be read-only 
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

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

# Wrapper around dictionary to make its keys callable attributes
class Args(dict):
    def __init__(self, *args, **kwargs):
        super(Args, self).__init__(*args, **kwargs)
        self.__dict__ = self
            
# Training function 
class Trainer:
    @ex.capture(prefix="train")
    def train(self, args):
        self.args = Args(args)  
        cudnn.benchmark = True
        # Fix seed for reproducibility
        torch.manual_seed(self.args.seed)  

        # Initialize solver with the defined arguments 
        self.solver = Solver(self.args)
        # Launch the training loop 
        results = self.solver.train()
        
        return results

# Functions to interact with seml (https://github.com/TUM-DAML/seml), an open source python package to interact with the slurm scheduling system
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print("get_experiment")
    experiment = Trainer()
    return experiment

@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = Trainer()
    return experiment.train()

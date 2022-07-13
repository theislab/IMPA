from torch.backends import cudnn
import torch

import seml
from sacred import Experiment
from sacred import SETTINGS

import sys

sys.path.insert(0, '../..')
# from compert.core.solver import Solver
from core.solver import Solver

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


#################### ARGS CLASS ####################

class Args(dict):
    def __init__(self, *args, **kwargs):
        super(Args, self).__init__(*args, **kwargs)
        self.__dict__ = self
            

#################### MAIN TRAINING FUNCTION ####################

class Trainer:
    @ex.capture(prefix="train")
    def train(self, args):
        # Fetch arguments 
        self.args = Args(args)  # Create argument class to access the args as attributes
        cudnn.benchmark = True
        torch.manual_seed(self.args.seed)  # Fix seed for reproducibility

        # Initialize solver with the defined arguments 
        self.solver = Solver(self.args)
        # Launch the training loop 
        results = self.solver.train()
        return results

# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print("get_experiment")
    experiment = Trainer()
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = Trainer()
    return experiment.train()

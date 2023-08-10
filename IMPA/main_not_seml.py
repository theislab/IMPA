import seml
import torch
from sacred import SETTINGS, Experiment
from torch.backends import cudnn
from IMPA.solver import Solver
import argparse
import yaml

# Wrapper around dictionary to make its keys callable attributes
class Args(dict):
    def __init__(self, *args, **kwargs):
        super(Args, self).__init__(*args, **kwargs)
        self.__dict__ = self
            
# Training function 
class Trainer:
    def train(self, args):
        self.args = Args(args["args"])  
        cudnn.benchmark = True
        # Fix seed for reproducibility
        torch.manual_seed(self.args.seed)  

        # Initialize solver with the defined arguments 
        self.solver = Solver(self.args)
        # Launch the training loop 
        results = self.solver.train()
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to the configuration file')

    args = parser.parse_args()

    # Access the path provided through the terminal
    path = args.path

    # Your code to process the file goes here
    args = yaml.safe_load(open(path))

    T = Trainer()
    
    T.train(args)
    
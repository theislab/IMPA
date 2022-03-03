import argparse
from training_utils import *
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform a training loop')
    parser.add_argument('--config_file', metavar='Configuration path', type=str, help='File with model configuration statistics', required = True)
    parser.add_argument('--mode', metavar='Train or test mode', type=str, help='Whether to use the model to train or test', required = True, choices = ['train', 'test'])    
    args = parser.parse_args()

    # Get configuration file 
    config = Config(config_path = args.config_file) 
    
    if args.mode == 'train':
        print(f'Training on {torch.cuda.device_count()} GPUs')
        # Setup the training loop 
        t = Trainer(config)
        t.train()
    
    else:
        print('Testing...')
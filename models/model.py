import torch
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm
import numpy as np
from training_utils import *
from metrics import *

class TemplateModel(nn.Module):
    def __init__(self):
        pass
    
    def save_checkpoints(self, epoch, metrics, losses, checkpoint_path):
        """
        Save the checkpoints to a checkpoint dict
        """
        new_checkpoint = dict()
        new_checkpoint['epoch'] = epoch
        new_checkpoint['model_state_dict'] = self.state_dict()
        for metric in metrics:
            new_checkpoint[metric] = metrics[metric]
        for loss in losses:
            new_checkpoint[loss] = losses[loss]
        

    def load_checkpoints(self):


# Inspired by https://github.com/clovaai/stargan-v2/blob/master/core/checkpoint.py

import os
import torch

class CheckpointIO:
    """
    Checkpoint class for saving model snapshots during training. 
    """
    def __init__(self, file_template, data_parallel=False, **kwargs):
        """
        Args:
            file_template (str): name of destination directory for the checkpoints.
            data_parallel (bool, optional): True if the model is wrapped in a torch.nn.DataParallel module. Defaults to False.
        """
        os.makedirs(os.path.dirname(file_template), exist_ok=True)
        self.file_template = file_template 
        self.module_dict = kwargs
        self.data_parallel = data_parallel

    def save(self, step):
        """
        Save the module checkpoints.

        Args:
            step (int): the iteration step at which the model is saved.
        """
        fname = self.file_template.format(step)
        print('Saving checkpoint into %s...' % fname)
        outdict = {}
        for name, module in self.module_dict.items():
            if self.data_parallel:
                outdict[name] = module.module.state_dict()
            else:
                outdict[name] = module.state_dict()  
        torch.save(outdict, fname)

    def load(self, step):
        """
        Load a checkpoint dictionary. 

        Args:
            step (int): the iteration step of the loaded model.
        """
        fname = self.file_template.format(step)
        assert os.path.exists(fname), fname + ' does not exist!'
        print('Loading checkpoint from %s...' % fname)
        if torch.cuda.is_available():
            print(fname)
            module_dict = torch.load(fname)
        else:
            module_dict = torch.load(fname, map_location=torch.device('cpu'))
        # Parametrise the modules 
        for name, module in self.module_dict.items():
            if self.data_parallel:
                module.module.load_state_dict(module_dict[name])
            else:
                module.load_state_dict(module_dict[name])

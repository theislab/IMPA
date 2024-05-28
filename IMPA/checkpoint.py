# Inspired by https://github.com/clovaai/stargan-v2/blob/master/core/checkpoint.py
import os
import torch

class CheckpointIO:
    """Checkpoint class for saving and loading model snapshots during training."""
    
    def __init__(self, file_template, data_parallel=False, **kwargs):
        """
        Initialize the CheckpointIO instance.
        
        Args:
            file_template (str): Template for the checkpoint file path.
            data_parallel (bool, optional): If True, the model is wrapped in a torch.nn.DataParallel module. Defaults to False.
            **kwargs: Arbitrary keyword arguments for module dictionary.
        """
        os.makedirs(os.path.dirname(file_template), exist_ok=True)
        self.file_template = file_template 
        self.module_dict = kwargs
        self.data_parallel = data_parallel

    def save(self, step):
        """Save the module checkpoints.
        
        Args:
            step (int): The iteration step at which the model is saved.
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
        """Load a checkpoint dictionary.
        
        Args:
            step (int): The iteration step of the model to be loaded.
        """
        fname = self.file_template.format(step)
        assert os.path.exists(fname), fname + ' does not exist!'
        print('Loading checkpoint from %s...' % fname)
        if torch.cuda.is_available():
            module_dict = torch.load(fname)
        else:
            module_dict = torch.load(fname, map_location=torch.device('cpu'))
        # Parameterize the modules
        for name, module in self.module_dict.items():
            if self.data_parallel:
                module.module.load_state_dict(module_dict[name])
            else:
                module.load_state_dict(module_dict[name])

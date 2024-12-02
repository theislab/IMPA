import torch
import numpy as np

def t2np(t, batch_dim=False):
    """
    Convert a PyTorch tensor to a NumPy array and normalize pixel values.

    Parameters:
        t (torch.Tensor): The PyTorch tensor to be converted.

    Returns:
        numpy.ndarray: The NumPy array representation of the tensor.
    """
    if not batch_dim:
        return ((t.permute(1, 2, 0) + 1) / 2).clamp(0, 1).cpu().numpy()
    else: 
        return ((t.permute(0, 2, 3, 1) + 1) / 2).clamp(0, 1).cpu().numpy()
    
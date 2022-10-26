import numpy as np
import torch
from scipy.spatial.distance import minkowski
from scipy.stats import ks_2samp


def dists(data): 
    """Compute intra-class distance

    Args:
        data (torch.tensor): a tensor of observations

    Returns:
        numpy.array: intra-class distances 
    """
    num = data.shape[0]
    data = data.reshape((num, -1))
    dist = []
    for i in tqdm(range(0,num-1)):
        for j in range(i+1,num):
            dist.append(minkowski(data[i],data[j]))
    return np.array(dist)

def dist_btw(a,b):
    """Compute between-class distance

    Args:
        a (numpy.array): images of domain 1 
        b (numpy.array): images of domain 2

    Returns:
        numpy.array: between-class distances 
    """
    a = a.reshape((a.shape[0], -1))
    b = b.reshape((b.shape[0], -1))
    dist = []
    for i in tqdm(range(a.shape[0])):
        for j in range(b.shape[0]):
            dist.append(minkowski(a[i],b[j]))
    return np.array(dist)


def LS(real,gen):  
    """Likeness score

    Args:
        real (numpy.array): tensor of real images
        gen (numpy.array): tensor of generated images 

    Returns:
        float: the Likeness score
    """
    # Compute intra-class distance 1
    dist_real = dists(real)
    # Compute intra-class distance 2  
    dist_gen = dists(gen)  
    # Compute between-clas distance 
    distbtw = dist_btw(real, gen)  
    
    # Kolmogorov-Smirnov 2-sample test    
    D_Sep_1, _ = ks_2samp(dist_real, distbtw)
    D_Sep_2, _ = ks_2samp(dist_gen, distbtw)
    
    # Likeness score
    return 1- np.max([D_Sep_1, D_Sep_2])  


def gpu_LS(real,gen):
    """GPU-accelerated Likeness Score

    Args:
        real (torch.tensor): tensor of real images
        gen (torch.tensor): tensor of generated images 

    Returns:
        float: the Likeness score 
    """
    t_gen = torch.from_numpy(gen)
    t_real = torch.from_numpy(real)

    # Compute intra-class distance 1
    dist_real = torch.cdist(t_real, t_real)  
    # Remove repeated entries
    dist_real = torch.flatten(torch.tril(dist_real, diagonal=-1))
    dist_real = dist_real[dist_real.nonzero()].flatten()  
    
    # Compute intra-class distance 1
    dist_gen = torch.cdist(t_gen, t_gen) 
    # Remove repeated entries
    dist_gen = torch.flatten(torch.tril(dist_gen, diagonal=-1)) 
    dist_gen = dist_gen[dist_gen.nonzero()].flatten()  
    
    # Compute between-class distance 
    distbtw = torch.cdist(t_gen, t_real) 
    distbtw = torch.flatten(distbtw)

    # Kolmogorov-Smirnov 2-sample test
    D_Sep_1, _ = ks_2samp(dist_real, distbtw)
    D_Sep_2, _ = ks_2samp(dist_gen, distbtw)

    # Likeness score
    return 1-np.max([D_Sep_1, D_Sep_2])  

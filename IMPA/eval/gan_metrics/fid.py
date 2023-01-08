import numpy as np
import torch
from scipy import linalg
import sys

sys.path.append('/home/icb/alessandro.palma/IMPA/imCPA/compert/eval/gan_metrics')
from inception import InceptionV3


def inception_activations(data_generator, model, dims=2048, custom_channels=None, use_cuda=True):
    """Compute the mean and average activations according to Inception V3

    Args:
        data_generator (torch.data.utils.DataLoader): data loader with images
        model (torch.nn.model): the model for feature extraction 
        dims (int, optional): number of dimensions. Defaults to 2048.
        custom_channels (list, optional): the channels used to compute the FID (in case more than 3). Defaults to None.
        use_cuda (bool, optional): use gpu or cpu. Defaults to True.

    Returns:
        tuple: mean and covariance of the image encodings computed for each image batch
    """
    device = 'cuda' if use_cuda==True else 'cpu'
    scores = []
    for batch in data_generator:
        batch_data = batch[0]
        # If more than 3 channels, select a subset of them 
        if batch_data.shape[1]>3:
            batch_data = batch_data[:,custom_channels, :, :]
        # Predict encodings with the model 
        pred = model(batch_data)[0] 
        # Append the scores for each batch
        scores.append(pred.view(-1, dims))
        
    # Concatenate scores vertically
    features = torch.cat(scores, dim=0).cpu().data.numpy()
    # return mu, sigma
    return np.mean(features, axis=0), np.cov(features, rowvar=False)


def cal_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Fréchet distance 

    Args:
        mu1 (torch.Tensor): average inception features dataset 1
        sigma1 (torch.Tensor): the covariance matrix from features of dataset 1
        mu2 (torch.Tensor): average inception features dataset 2
        sigma2 (torch.Tensor): average inception features dataset 2
        eps (int, optional): small constant to avoid zero division. Defaults to 1e-6.

    Raises:
        ValueError: presence of imaginary component due to numerical errors 

    Returns:
        float: the Fréchet distance computed from mu1, sigma1, mu2, sigma2
    """
    # Turn the means and covariance matrices into numpy arrays
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2

    # Calculate Frechét inception distance
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # If singular values, add small constant 
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Extract real component if the squared root of the matrix is complex
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    # Trace of the 
    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def cal_fid(data1, data2, dims, use_cuda, custom_channels=None):
    """Calculate the Fréchet inception distance score

    Args:
        data1 (torch.data.utils.DataLoader): the first dataset to compare
        data2 (torch.data.utils.DataLoader): the second dataset to compare
        dims (int): number of dimensions
        use_cuda (bool): use gpu or cpu
        custom_channels (list, optional): channels to compare between the datasets. Defaults to None.

    Returns:
        float: the Fréchet inception distance between data1 and data2
    """
    # Select last pooling layer
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    
    # Whether the model acts on CPU or GPU
    if use_cuda:
        model.cuda()
    model.eval()

    # Compute mean and standard deviation of the two distributions (real vs fake) 
    m1, s1 = inception_activations(data1, model, dims, custom_channels, use_cuda)
    m2, s2 = inception_activations(data2, model, dims, custom_channels, use_cuda)
    fid_value = cal_frechet_distance(m1, s1, m2, s2)

    return fid_value

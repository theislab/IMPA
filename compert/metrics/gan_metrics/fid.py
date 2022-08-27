import numpy as np
from scipy import linalg

import torch
from torch.nn.functional import adaptive_avg_pool2d

from .inception import InceptionV3

def inception_activations(data_generator, model, dims=2048, custom_channels=None, use_cuda=True):
    """
    Derive the activations from the inceptionV3 network
    """
    device = 'cuda' if use_cuda==True else 'cpu'
    scores = []
    for batch in data_generator:
        batch_data = batch[0]
        if batch_data.shape[1]>3:
            batch_data = batch_data[:,custom_channels, :, :]
        pred = model(batch_data)[0] 
        # Append the scores for each batch
        scores.append(pred.view(-1, dims))
        
    # Concatenate scores vertically
    acts = torch.cat(scores, dim=0).cpu().data.numpy()
    # return mu, sigma
    return np.mean(acts, axis=0), np.cov(acts, rowvar=False)


def cal_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def cal_fid(data1, data2, dims, use_cuda, custom_channels=None):
    """Calculates the FID of two data generator.
    Params:
    -- data1   : generator of data one.
    -- data2   : generator of data two.
    -- use_cuda: use cuda or not.
    -- dims.   : feature dimensionality 2048.
    """
    # Model to compute the inception score 
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

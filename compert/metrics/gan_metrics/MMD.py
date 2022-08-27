import torch
import torch.nn as nn

import torch
min_var_est = 1e-8


def multiscale_rbf(X, Y, sigma_list):
    assert(X.size(0) == Y.size(0))
    n = X.shape[0]
    XY = torch.cat([X, Y], dim=0)
    XY1 = XY.unsqueeze(0).expand(int(XY.size(0)), int(XY.size(0)), int(XY.size(1)))
    XY2 = XY.unsqueeze(1).expand(int(XY.size(0)), int(XY.size(0)), int(XY.size(1)))
    L2_distance = ((XY1-XY2)**2).sum(2) 

    K=0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * L2_distance)
    return K[:n, :n], K[:n, n:], K[n:, n:]


def mmd_rbf(X, Y, sigma_list, biased=True):
    n = X.shape[0]
    K_XX, K_XY, K_YY = multiscale_rbf(X, Y, sigma_list)
    diag_X = torch.diag(K_XX)                      
    diag_Y = torch.diag(K_YY)   

    if not biased:
        K_XX = K_XX.sum(1) - diag_X
        K_YY = K_YY.sum(1) - diag_Y
        return 1/(n*(n-1))*K_XX.sum() - 2/(n*n)*K_XY.sum() + 1/(n*(n-1))*K_YY.sum()
    else:
        K_XX = K_XX.sum(1) 
        K_YY = K_YY.sum(1)
        return 1/(n**2)*K_XX.sum() - 2/(n*n)*K_XY.sum() + 1/(n**2)*K_YY.sum()
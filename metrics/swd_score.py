import torch
import torch.nn as nn
import numpy as np

# From https://github.com/facebookresearch/pytorch_GAN_zoo/blob/main/models/metrics/laplacian_swd.py
def sample_patches(X, patchSize, nPatches):
    """
    Sample random patches from a batch
    ---------------------------
    X: minibatch of simensions B x C x H x W
    patchSize: the size of the patches 
    nPatch: The number of patches per batch element 
    
    Return: B x Patch_no x Patch_size x Patch_size tensor
    """
    # The dimensions of the minibatch 
    S = X.size()

    # Maximum attainable value per spatial dimension of an image 
    maxX = S[2] - patchSize
    maxY = S[3] - patchSize
    
    # Compute  and Y ranges of 128 patches for each image in the batch 
    baseX = torch.arange(0, patchSize, dtype=torch.long).expand(S[0] * nPatches,
                                                                patchSize) \
        + torch.randint(0, maxX, (S[0] * nPatches, 1), dtype=torch.long)  # B*P x P_size

    baseY = torch.arange(0, patchSize, dtype=torch.long).expand(S[0] * nPatches,
                                                                patchSize) \
        + torch.randint(0, maxY, (S[0] * nPatches, 1), dtype=torch.long)   # B*P x P_size

    # Obtain marginal patches for X and Y 
    baseX = baseX.view(S[0], nPatches, 1, patchSize).expand(
        S[0], nPatches, patchSize, patchSize) 
    baseY = S[2] * baseY.view(S[0], nPatches, patchSize, 1)  #Two pixels aligned on the same column are distantiated by 64 pixels 
    baseY = baseY.expand(S[0], nPatches, patchSize, patchSize)

    # Sum the patches of X and Y to get a progression on both directions
    coords = baseX + baseY  

    # Fix the coordinates of the pixels across channels and select them from the minibatc
    coords = coords.view(S[0], nPatches, 1, patchSize, patchSize).expand(
        S[0], nPatches, S[1], patchSize, patchSize)
    C = torch.arange(0, S[1], dtype=torch.long).view(
        1, S[1]).expand(nPatches * S[0], S[1])*S[2]*S[3]  # The indexes representing each channel are separated by 64^2 elements
    coords = C.view(S[0], nPatches, S[1], 1, 1) + coords
    coords = coords.view(-1)  # Linearized coordinates to index

    return (X.contiguous().view(-1)[coords]).view(-1, S[1], patchSize, patchSize)


# Laplacian pyramid up-down sampling
def downsample(X, conv):
    """
    Input a minibatch X and a convolution operator.
    ---------------------------
    X: minibatch of simensions B x C x H x W
    conv: convoutional layer following Laplacian pyramid 
    """
    x = torch.nn.ReflectionPad2d(2)(X)
    return conv(x)[:, :, ::2, ::2].detach()


def upsample(X, conv):
    """
    Input a minibatch X and a convolution operator.
    ---------------------------
    X: minibatch of simensions B x C x H x W
    conv: convoutional layer following Laplacian pyramid 
    """
    S = X.size()
    res = torch.zeros((S[0], S[1], S[2] * 2, S[3] * 2),
                      dtype=X.dtype).to(X.device)
    res[:, :, ::2, ::2] = X
    res = torch.nn.ReflectionPad2d(2)(res)
    return conv(res).detach()


def swd(A, B, dir_repeats, dirs_per_repeat):
    """
    The original implementation of the SWD as described by https://arxiv.org/abs/1710.10196
    -----------------------------------
    A, B: Tensors of reference that must be compared
    dir_repeats: the number of directions used to perform tje 
    """
    projection_outcomes = []
    for repeat in range(dir_repeats):
        # Sample a direction
        dirs = torch.randn(A.shape[1], dirs_per_repeat,
                           device=A.device, dtype=torch.float32)
        # Normalize the direction by the squared L2 norm of the vector
        dirs /= torch.sqrt(torch.sum(dirs*dirs, 0, keepdim=True))
        # Project the rows of A and B onto the directions of interest        
        projA = torch.matmul(A, dirs)
        projB = torch.matmul(B, dirs)
        # Sort neighborhood projections for each direction
        projA = torch.sort(projA, dim=0)[0]
        projB = torch.sort(projB, dim=0)[0]
        # Calculate pointwise wasserstein distances
        dists = torch.abs(projA - projB)
        # Average over neighborhoods and directions
        projection_outcomes.append(torch.mean(dists).item())
    # Average counts 
    return sum(projection_outcomes) / float(len(projection_outcomes))

def finalize_descriptors(desc):
    """
    Normalize input patches as suggested in https://arxiv.org/abs/1710.10196
    """
    if isinstance(desc, list):
        desc = torch.cat(desc, axis=0)
    assert desc.ndim == 4  # (neighborhood, channel, height, width)
    desc -= torch.mean(desc, axis=(0, 2, 3), keepdims=True)
    desc /= torch.std(desc, axis=(0, 2, 3), keepdims=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc


# -------------------------------------------------------------------------------
# Class to store and execute the computation of the SWD score. The metrics 
# are collected on the patches obtained from the Laplacian pyramid 
# -------------------------------------------------------------------------------

class LaplacianSWDMetric:
    def __init__(self,
                 patchSize,
                 patchNumber,
                 depthPyramid):
        """
        Args:
            patchSize (int): side length of each patch to extract
            patchNumber (int): number of patches to extract at each level
                                    of the pyramid
            depthPyramid (int): depth of the laplacian pyramid
        """
        self.patchSize = patchSize  # The side length of the patch
        self.patchNumber = patchNumber 
        self.depthPyramid = depthPyramid

        # Initialize the lists containing the descriptors of both reference and target at different resolutions 
        self.descriptorsRef = [[] for x in range(depthPyramid)]
        self.descriptorsTarget = [[] for x in range(depthPyramid)]

        # Will be initialized with a convolution  
        self.convolution_down = None

    
    def update_with_mini_batch(self, ref, target):
        """
        Extract and store descriptors from the current minibatch
        --------------------------------------------------------
        ref (tensor): reference batch BxCxHxW
        target (tensor): target batch BxCxHxW
        """
        target = target.to(ref.device)
        modes = [(ref, self.descriptorsRef), (target, self.descriptorsTarget)]

        assert(ref.size() == target.size())

        # Initialize convolutional filter for the Laplacian pyramid 
        if not self.convolution_down:
            self.init_convolution(ref.device)
        
        for item, dest in modes:
            pyramid = self.generate_laplacian_pyramid(item, self.depthPyramid)  # List with as many pyramids as levels 
            for scale in range(self.depthPyramid):
                # For each minibatch, append a different patch object
                dest[scale].append(sample_patches(pyramid[scale],
                                                              self.patchSize,
                                                              self.patchNumber))

    
    def generate_laplacian_pyramid(self, X, num_levels):
        """
        Build the Laplacian pyramids corresponding to the current minibatch.
        Args:
            X (tensor): B x C x H x W, input batch
            num_levels (int): number of levels of the pyramids
        """
        pyramid = [X]
        for i in range(1, num_levels):
            # Append downsampled pyramid and subtract the upsampled version of it to the original one 
            pyramid.append(downsample(pyramid[-1], self.convolution_down))
            pyramid[-2] -= upsample(pyramid[-1], self.convolution_up)
        return pyramid  


    def get_score(self):
        """
        Output the SWD distance between reference and destination patch distributions 
        """
        output = []

        # Normalize and the descriptors and linearize them to dimension n_batches x -1
        descTarget = [finalize_descriptors(d) for d in self.descriptorsTarget]
        del self.descriptorsTarget

        descRef = [finalize_descriptors(d) for d in self.descriptorsRef]
        del self.descriptorsRef

        for scale in range(self.depthPyramid):
            distance = swd(
                descTarget[scale], descRef[scale], 4, 128)  # 512 projections 
            output.append(distance*1e3) 
        
        del descRef, descTarget 

        return np.mean(output)

    def reconstruct_laplacian_pyramid(self, pyramid):
        """
        Given a Laplacian pyramid, reconstruct the corresponding minibatch
        Returns:
            A list L of tensors NxCxWxD, where L[i] represents the pyramids of
            the batch for the ith scale
        """
        X = pyramid[-1]
        for level in pyramid[-2::-1]:
            X = upsample(X, self.convolution_up) + level
        return X


    def init_convolution(self, device):
        """
        Initialize the convolution for the Laplacian pyramid
        """
        gaussianFilter = torch.tensor([
            [1, 4,  6,  4,  1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4,  6,  4,  1]], dtype=torch.float) / 256.0

        self.convolution_down = nn.Conv2d(5, 5, (5, 5))
        self.convolution_up = nn.Conv2d(5, 5, (5, 5))

        self.convolution_down.weight.data.fill_(0)
        self.convolution_up.weight.data.fill_(0)

        # Downsampling 
        self.convolution_down.weight.data[0][0] = gaussianFilter
        self.convolution_down.weight.data[1][1] = gaussianFilter
        self.convolution_down.weight.data[2][2] = gaussianFilter
        self.convolution_down.weight.data[3][3] = gaussianFilter
        self.convolution_down.weight.data[4][4] = gaussianFilter

        # Upsampling 
        self.convolution_up.weight.data[0][0] = gaussianFilter*4
        self.convolution_up.weight.data[1][1] = gaussianFilter*4
        self.convolution_up.weight.data[2][2] = gaussianFilter*4
        self.convolution_up.weight.data[3][3] = gaussianFilter*4
        self.convolution_up.weight.data[4][4] = gaussianFilter*4

        self.convolution_down.weight.requires_grad = False
        self.convolution_up.weight.requires_grad = False

        self.convolution_down = self.convolution_down.to(device)
        self.convolution_up = self.convolution_up.to(device)


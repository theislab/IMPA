import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from munch import Munch

# Inspired by https://github.com/clovaai/stargan-v2/blob/master/core/model.py

class ResBlk(nn.Module):
    """
    Basic residual block with convolutions 
    """
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        """Build core network layers

        Args:
            dim_in (int): input dimensionality 
            dim_out (int): output dimensionality 
        """
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        """Shortcut connection in the residual framework 

        Args:
            x (torch.Tensor): The input image 

        Returns:
            torch.Tensor: Input processed by the skip connection  
        """
        # If the shortcut is to be learned 
        if self.learned_sc:
            x = self.conv1x1(x)
        # If downsampling is to be performed 
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        """Residual connection 

        Args:
            x (torch.Tensor): The input image 

        Returns:
            torch.Tensor: Input processed by the residual connection  
        """
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        # Return output normalized to unit variance
        return x / math.sqrt(2)  


class AdaIN(nn.Module):
    """Adaptive instance normalization 
    """
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s, basal=False):
        if not basal:
            h = self.fc(s)
            h = h.view(h.size(0), h.size(1), 1, 1)
            gamma, beta = torch.chunk(h, chunks=2, dim=1)
            return (1 + gamma) * self.norm(x) + beta
        else:
            return self.norm(x)


class AdainResBlk(nn.Module):
    """Decoding block with AdaIN
    """
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.actv = actv  
        self.upsample = upsample 
        self.learned_sc = dim_in != dim_out  
        self._build_weights(dim_in, dim_out, style_dim) 

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        """Build core network layers

        Args:
            dim_in (int): input dimension
            dim_out (int): output dimension 
            style_dim (int, optional): style dimension. Defaults to 64.
        """
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        # Adaptive normalizations 
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        """Shortcut connection in the residual framework 

        Args:
            x (torch.Tensor): The input image 

        Returns:
            torch.Tensor: Input processed by the skip connection  
        """
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s, basal=False):
        """Residual connection 

        Args:
            x (torch.Tensor): The input image 

        Returns:
            torch.Tensor: Input processed by the residual connection  
        """
        x = self.norm1(x, s, basal)  # AdaIn incorporation 
        x = self.actv(x)  # Activation function 
        if self.upsample: 
            x = F.interpolate(x, scale_factor=2, mode='nearest')  # Upsample by interpolation 
        x = self.conv1(x)
        x = self.norm2(x, s, basal)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s, basal=False):
        # Apply residual connection to input and condition
        out = self._residual(x, s, basal) + self._shortcut(x)
        # out = self._residual(x, s, basal) 
        return out / math.sqrt(2)
        # return out
    

class Generator(nn.Module):
    """The autoencoding model used for the transformation 
    """
    def __init__(self, img_size=96, style_dim=64, max_conv_dim=512, in_channels=3, dim_in=64):
        super().__init__()
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(in_channels, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        # First block from raw images 
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, in_channels, 1, 1, 0))

        # For images of size 96x96 you have 3 convolutions and reach spatial dimension of 12x12 on conv filters
        repeat_num = math.ceil(np.log2(img_size)) - 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, upsample=True))  
            dim_in = dim_out

        # The bottleneck is residual network 
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim))

    def forward(self, x, s, basal=False):
        # First convolution with RGB 
        x = self.from_rgb(x)
        for block in self.encode:    
            x = block(x)
        # Save the latent as a return value 
        z = x.clone()
        for block in self.decode:
            x = block(x, s, basal)  # if basal is true we don't condition on the perturbation
        return z, self.to_rgb(x)
    
    def encode_single(self, x):
        """Encode a single observation 

        Args:
            x (torch.Tensor): input images 

        Returns:
            torch.Tensor: processed output images 
        """
        # x = self.from_rgb(x)
        x = self.from_rgb(x)
        for block in self.encode:    
            x = block(x)
        return x

    def decode_single(self, x, s, basal=False):
        """Decode a single observation  

        Args:
            x (torch.Tensor): Input latent image 
            s (torch.Tensor): Conditioning embeddings 
            basal (bool, optional): Whether condition the decoding or not. Defaults to False.

        Returns:
            torch.Tensor: Generated images 
        """
        for block in self.decode:
            x = block(x, s, basal) 
        return self.to_rgb(x) 
    

class MappingNetwork(nn.Module):
    """Linear projection for raw embeddings 
    """
    def __init__(self, latent_dim=160, style_dim=64, hidden_dim=512, num_layers=4):
        super().__init__()
        in_dim = latent_dim 
        out_dim = hidden_dim 
        layers = []
        
        for i in range(num_layers):
            # Style dimension 
            out_dim = hidden_dim if i < num_layers-1 else style_dim
            layers += [nn.Linear(in_dim, out_dim)]
            in_dim = out_dim
            if i < num_layers-1:
                layers += [nn.ReLU()]                
        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        h = self.layers(z) 
        return h


class StyleEncoder(nn.Module):
    """Encoder from images to style vector 
    """
    def __init__(self, img_size=96, style_dim=64, max_conv_dim=512, in_channels=3, dim_in=64):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(in_channels, dim_in, 3, 1, 1)]

        # For 96x96 image this downsamples till 3x3 spatial dimension 
        repeat_num = math.ceil(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        # Downsamples to spatial dimensionality of 1 
        blocks += [nn.Conv2d(dim_out, dim_out, 3, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.conv = torch.nn.Sequential(*blocks)

        self.linear = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        # Apply shared layer and linearize 
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z_style = self.linear(h)
        return z_style


class Discriminator(nn.Module):
    """Discriminator network for the GAN model 
    """
    def __init__(self, img_size=96, num_domains=2, max_conv_dim=512, in_channels=3, dim_in=64, multi_task=True):
        super().__init__()
        self.multi_task = multi_task
        blocks = []
        blocks += [nn.Conv2d(in_channels, dim_in, 3, 1, 1)]

        repeat_num = math.ceil(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2, inplace=True)]
        blocks += [nn.Conv2d(dim_out, dim_out, 3, 1, 0)]
        blocks += [nn.LeakyReLU(0.2, inplace=True)]
        # Final convolutional layer that points to the number of domains 
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.conv = nn.Sequential(*blocks)

    def forward(self, x, y):
        # Apply the network on X 
        out = self.conv(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains) 
        if self.multi_task:
            idx = torch.LongTensor(range(y.size(0))).to(y.device)
            out = out[idx, y]  # (batch)
        return out
    

def build_model(args):
    # Generator autoencoder
    generator = nn.DataParallel(Generator(args.img_size, args.style_dim, in_channels=args.n_channels, dim_in=args.dim_in))
    
    # Style encoder 
    style_encoder = nn.DataParallel(StyleEncoder(args.img_size, args.style_dim, in_channels=args.n_channels, dim_in=args.dim_in))

    # Discriminator network 
    discriminator = nn.DataParallel(Discriminator(args.img_size, args.num_domains, in_channels=args.n_channels, dim_in=args.dim_in))
    
    # The rdkit embeddings can be collected together with noise 
    if args.stochastic:
        input_dim = args.latent_dim + args.z_dimension
    else:
        input_dim = args.latent_dim 
    
    # Mapping network
    mapping_network = nn.DataParallel(MappingNetwork(input_dim, args.style_dim, hidden_dim=512, num_layers=args.num_layers_mapping_net))

    # Dictionary with the modules 
    nets = Munch(generator=generator,
            style_encoder=style_encoder, 
            discriminator=discriminator, 
            mapping_network=mapping_network)
    
    return nets

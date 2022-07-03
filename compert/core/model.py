import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

##################### BASIC RESIDUAL BLOCK WITH CONVOLUTIONS #####################

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        # Employed activation function 
        self.actv = actv
        # Normalize true or false
        self.normalize = normalize
        # Downsample in the resnet block 
        self.downsample = downsample
        # If the shortcut should be learnt
        self.learned_sc = dim_in != dim_out
        # Build the networks used in the residual block
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        # If the shortcut is to be learned 
        if self.learned_sc:
            x = self.conv1x1(x)
        # If downsampling is to be performed 
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        # Blocks of normalization - activation - convolution - pool
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


##################### ADAPTIVE INSTANCE NORMALIZATION #####################

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s, basal=False):
        if not basal:
            # Condition is first submitted to a linear layer 
            h = self.fc(s)
            # Unsqueeze the spatial dimension, break into scale and bias and broadcast
            h = h.view(h.size(0), h.size(1), 1, 1)
            gamma, beta = torch.chunk(h, chunks=2, dim=1)
            return (1 + gamma) * self.norm(x) + beta
        else:
            return self.norm(x)


##################### DECODING RESIDUAL BLOCK WITH INSTANCE NORMALIZATION #####################

class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.actv = actv  # Activation 
        self.upsample = upsample  # Upsampling method 
        self.learned_sc = dim_in != dim_out  # Learned shortcut 
        self._build_weights(dim_in, dim_out, style_dim) 

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        # Non-downsizing, depth-preserving convolutions
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        # Adaptive normalizations 
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        # Shortcut
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s, basal=False):
        x = self.norm1(x, s, basal)  # AdaIn incorporation of the condition 
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
        out = self._residual(x, s, basal)
        return out


##################### GENERATOR (Autoencoder structure) #####################

class Generator(nn.Module):
    def __init__(self, img_size=96, style_dim=64, max_conv_dim=512, in_channels=3):
        super().__init__()
        # Initial dimension of the feature map
        dim_in = 64
        self.img_size = img_size
        # First convolutional layer 
        self.from_rgb = nn.Conv2d(in_channels, dim_in, 3, 1, 1)
        # Encoding modules and decoding modules
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        # Last convolutional block of decoder 
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, in_channels, 1, 1, 0))

        # For images of size 96x96 you have 3 convolutions and reach spatial dimension of 12x12 on conv filters
        repeat_num = math.ceil(np.log2(img_size)) - 4
        for _ in range(repeat_num):
            # Dimensionality of the output on the feature map 
            dim_out = min(dim_in*2, max_conv_dim)
            # Build residual blocks in the encoder 
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            # Build the conditioned residual blocks with adaptive instance normalization in the decoder 
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, upsample=True))  # stack-like
            dim_in = dim_out

        # The bottleneck is made of residual network 
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
            x = block(x, s, basal)  # If basal is true we will have absence of conditioning 
        return z, self.to_rgb(x)
    
    def encode_single(self, x):
        x = self.from_rgb(x)
        for block in self.encode:    
            x = block(x)
        return x

    def decode_single(self, x, s, basal=False):
        for block in self.decode:
            x = block(x, s, basal)  # If basal is true we will have absence of conditioning 
        return self.to_rgb(x) 


##################### MAPPING NETWORK FROM LATENT TO STYLE VECTOR #####################


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=160, style_dim=64, hidden_dim=512, num_layers=4):
        super().__init__()
        # Single neural network starting from the rdkit embeddings 
        in_dim = latent_dim 
        out_dim = hidden_dim 
        layers = []
        
        for i in range(num_layers):
            # Style dimension 
            out_dim = hidden_dim if i < num_layers-1 else style_dim
            # Linear layer
            layers += [nn.Linear(in_dim, out_dim)]
            # Increase in_dim
            in_dim = out_dim
            if i < num_layers-1:
                layers += [nn.ReLU()]                
        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        h = self.layers(z) 
        return h


##################### ENCODER FROM IMAGE TO STYLE #####################


class StyleEncoder(nn.Module):
    def __init__(self, img_size=96, style_dim=64, num_domains=2, max_conv_dim=512, in_channels=3):
        super().__init__()
        # Encoder for the style applied to the input data downsampling to the style vector dimension 
        dim_in = 64
        blocks = []
        blocks += [nn.Conv2d(in_channels, dim_in, 3, 1, 1)]

        # For 96x96 image this downsamples till 3x3 spatial dimension 
        repeat_num = math.ceil(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 3, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        # To the lower convolutional layer we attach a linear layer pointing to each domain
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        # Apply shared layer and linearize 
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        # Once again, select the right style vectors 
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s

class StyleEncoderSingle(nn.Module):
    def __init__(self, img_size=96, style_dim=64, max_conv_dim=512, in_channels=3):
        super().__init__()
        # Encoder for the style applied to the input data downsampling to the style vector dimension 
        dim_in = 64
        blocks = []
        blocks += [nn.Conv2d(in_channels, dim_in, 3, 1, 1)]

        # For 96x96 image this downsamples till 3x3 spatial dimension 
        repeat_num = math.ceil(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
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


##################### MULTI-TASK DISCRIMINATOR FOR TRUE/FALSE #####################


class Discriminator(nn.Module):
    def __init__(self, img_size=96, num_domains=2, max_conv_dim=512, in_channels=3):
        super().__init__()
        dim_in = 64
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
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        # Apply the network on X 
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out


##################### CONVOLUTIONAL DISENTANGLEMENT CLASSIFIER #####################


class LeakyReLUConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, norm='Instance'):
        super().__init__()
        model = []
        model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        # Normalization layer, can be instance or batch
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(out_channels, affine=True)]
        if norm == 'Batch':
            model += [nn.BatchNorm2d(out_channels)]
        # Leaky ReLU loss with default parameters
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class DisentanglementClassifier(nn.Module):
    def __init__(self, init_dim=12, init_fm=256, out_fm=64, num_outputs=2):
        super().__init__()
        self.init_dim = init_dim  # Spatial dimension
        self.init_fm = init_fm  # Input feature maps
        self.out_fm = out_fm  # Output feature maps 
        self.num_outpus = num_outputs  # Number of classes for the classification 

        # First number of feature maps 
        in_fm = self.init_fm 

        # For dimension 12 
        depth = int(np.around(np.log2(self.init_dim)))-2
        # Layers of LeakyReLU - Convolution 
        model = []
        for i in range(depth):
            model += [LeakyReLUConv2d(in_fm, out_fm, kernel_size=3, stride=2, padding=1)]
            in_fm = out_fm
            out_fm = out_fm*2
        
        model += [torch.nn.Conv2d(in_fm, self.num_outpus, 3)]
        # Compile model 
        self.conv = nn.Sequential(*model)

    def forward(self, x):
        out = self.conv(x)
        return out.view(out.shape[0], -1)


##################### GENERAL FUNCTION BUILDING THE NETS #####################

def build_model(args):
    # Generator autoencoder
    generator = nn.DataParallel(Generator(args.img_size, args.style_dim, in_channels=args.n_channels))

    # Style encoder for the images 
    if not args.single_style:
        style_encoder = nn.DataParallel(StyleEncoder(args.img_size, args.style_dim, args.num_domains, in_channels=args.n_channels))
    else:
        style_encoder = nn.DataParallel(StyleEncoderSingle(args.img_size, args.style_dim, in_channels=args.n_channels))

    # Discriminator network 
    discriminator = nn.DataParallel(Discriminator(args.img_size, args.num_domains, in_channels=args.n_channels))

    nets = Munch(generator=generator,
            style_encoder=style_encoder,
            discriminator=discriminator)
    
    # RDKit can be encoded or left pure
    if args.encode_rdkit:  
        # The rdkit embeddings can be collected together with noise 
        input_dim = args.latent_dim if not args.stochastic else args.latent_dim + args.z_dimension
        mapping_network = nn.DataParallel(MappingNetwork(input_dim, args.style_dim, hidden_dim=512, num_layers=args.num_layers_mapping_net))
        nets['mapping_network'] = mapping_network

    return nets

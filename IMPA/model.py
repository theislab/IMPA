import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from munch import Munch

# Inspired by https://github.com/clovaai/stargan-v2/blob/master/core/model.py

################## Building blocks ##################

class ResBlk(nn.Module):
    """
    Basic residual block with convolutions.
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
        """Build core network layers.

        Args:
            dim_in (int): Input dimensionality.
            dim_out (int): Output dimensionality.
        """
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        """Shortcut connection in the residual framework.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: Input processed by the skip connection.
        """
        # If the shortcut is to be learned.
        if self.learned_sc:
            x = self.conv1x1(x)
        # If downsampling is to be performed.
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        """Residual connection.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: Input processed by the residual connection.
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
        # Return output normalized to unit variance.
        return x / math.sqrt(2)  


class AdaIN(nn.Module):
    """Adaptive Instance Normalization (AdaIN) module.
    
    This layer modulates feature maps based on style input.
    """
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s, basal=False):
        """Forward pass for AdaIN.

        Args:
            x (torch.Tensor): Content input features.
            s (torch.Tensor): Style input features.
            basal (bool): If True, apply only normalization without style modulation.

        Returns:
            torch.Tensor: Modulated or normalized output.
        """
        if not basal:
            h = self.fc(s)
            h = h.view(h.size(0), h.size(1), 1, 1)
            gamma, beta = torch.chunk(h, chunks=2, dim=1)
            return (1 + gamma) * self.norm(x) + beta
        else:
            return self.norm(x)

class AdainResBlk(nn.Module):
    """Decoding block with AdaIN, combining residual connections and adaptive instance normalization.
    """
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.actv = actv  # Activation function to use.
        self.upsample = upsample  # Whether to upsample the input.
        self.learned_sc = dim_in != dim_out  # Determine if a learned skip connection is needed.
        self._build_weights(dim_in, dim_out, style_dim)  # Build the core layers.

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        """Build core network layers.

        Args:
            dim_in (int): Input dimension.
            dim_out (int): Output dimension.
            style_dim (int, optional): Style dimension. Defaults to 64.
        """
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        # Adaptive instance normalization layers.
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        """Shortcut connection in the residual framework.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: Input processed by the skip connection.
        """
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')  # Upsample the input if needed.
        if self.learned_sc:
            x = self.conv1x1(x)  # Apply 1x1 convolution if dimensions differ.
        return x

    def _residual(self, x, s, basal=False):
        """Residual connection.

        Args:
            x (torch.Tensor): The input image.
            s (torch.Tensor): Style input for AdaIN.
            basal (bool): If True, bypass style modulation.

        Returns:
            torch.Tensor: Input processed by the residual connection.
        """
        x = self.norm1(x, s, basal)  # Apply AdaIN normalization.
        x = self.actv(x)  # Apply activation.
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')  # Upsample if required.
        x = self.conv1(x)
        x = self.norm2(x, s, basal)  # Apply second AdaIN normalization.
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s, basal=False):
        """Forward pass for the AdainResBlk.

        Args:
            x (torch.Tensor): Input tensor.
            s (torch.Tensor): Style tensor for AdaIN.
            basal (bool): If True, apply only normalization without style modulation.

        Returns:
            torch.Tensor: Processed output with residual connection.
        """
        out = self._residual(x, s, basal) + self._shortcut(x)  # Combine residual and shortcut paths.
        return out / math.sqrt(2)  # Normalize output.

################## Generator (IMPA's decoder) ##################

class Generator(nn.Module):
    """The autoencoding model used for the image transformation process."""
    def __init__(self, img_size=96, style_dim=64, max_conv_dim=512, in_channels=3, dim_in=64):
        super().__init__()
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(in_channels, dim_in, 3, 1, 1)  # Initial RGB processing layer.
        self.encode = nn.ModuleList()  # Encoder block list.
        self.decode = nn.ModuleList()  # Decoder block list.
        
        # Final RGB output layer.
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, in_channels, 1, 1, 0)
        )

        # Calculate the number of convolution layers needed to reduce spatial dimensions.
        repeat_num = math.ceil(np.log2(img_size)) - 4

        # Encoder and corresponding decoder construction.
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True)
            )
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, upsample=True)  # AdaIN only in the decoder
            )
            dim_in = dim_out

        # Bottleneck consisting of residual blocks.
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True)
            )
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim)
            )

    def forward(self, x, s, basal=False):
        """Forward pass for the Generator.

        Args:
            x (torch.Tensor): Input image tensor.
            s (torch.Tensor): Style tensor for AdaIN conditioning.
            basal (bool, optional): Whether to bypass style conditioning. Defaults to False.

        Returns:
            torch.Tensor: Latent encoding and the final generated image.
        """
        # Initial processing with RGB.
        x = self.from_rgb(x)
        
        # Pass through encoder blocks.
        for block in self.encode:
            x = block(x)
        
        # Save the latent code.
        z = x.clone()
        
        # Pass through decoder blocks with possible style conditioning.
        for block in self.decode:
            x = block(x, s, basal)
        
        # Return the latent code and the RGB decoded image.
        return z, self.to_rgb(x)

    def encode_single(self, x):
        """Encode a single observation.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Encoded latent representation.
        """
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        return x

    def decode_single(self, x, s, basal=False):
        """Decode a single observation from its latent representation.

        Args:
            x (torch.Tensor): Latent tensor.
            s (torch.Tensor): Style tensor for AdaIN conditioning.
            basal (bool, optional): Whether to bypass style conditioning. Defaults to False.

        Returns:
            torch.Tensor: Final generated image.
        """
        for block in self.decode:
            x = block(x, s, basal)
        return self.to_rgb(x)
    
################## Perturbation encoding component ##################

class MappingNetworkSingleStyle(nn.Module):
    """Linear projection network to map raw embeddings to style embeddings. Works with single shared style space."""
    def __init__(self, latent_dim=160, style_dim=64, hidden_dim=512, num_layers=4):
        super().__init__()
        in_dim = latent_dim  # Input dimensionality (raw embedding size).
        out_dim = hidden_dim  # Initial hidden dimension.
        layers = []

        # Build a multi-layer perceptron (MLP) with ReLU activations between layers.
        for i in range(num_layers):
            # Set the output dimension to `hidden_dim` for all layers except the last.
            out_dim = hidden_dim if i < num_layers - 1 else style_dim
            layers += [nn.Linear(in_dim, out_dim)]  # Fully connected layer.
            in_dim = out_dim
            if i < num_layers - 1:
                layers += [nn.ReLU()]  # ReLU activation for all layers except the final one.
                
        self.layers = nn.Sequential(*layers)  # Stack all layers.

    def forward(self, z):
        """Forward pass for the mapping network.

        Args:
            z (torch.Tensor): Input latent vector.

        Returns:
            torch.Tensor: Style embedding after linear projection.
        """
        h = self.layers(z)  # Pass the input through the network.
        return h

class MappingNetworkMultiStyle(nn.Module):
    """Mapping network for multi-domain style embeddings, consisting of a shared 
    and domain-specific projection for the style embeddings."""
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        # Shared layers common to all domains.
        layers = []
        layers += [nn.Linear(latent_dim, 512)]  # First fully connected layer.
        layers += [nn.ReLU()]  # Activation after the first layer.
        for _ in range(3):
            layers += [nn.Linear(512, 512)]  # Add more fully connected layers.
            layers += [nn.ReLU()]  # ReLU activation for hidden layers.
        self.shared = nn.Sequential(*layers)  # Shared network part.

        # Unshared domain-specific layers for each domain.
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(
                nn.Linear(512, 512),  # First fully connected layer for domain.
                nn.ReLU(),            # ReLU activation.
                nn.Linear(512, 512),  # Second fully connected layer for domain.
                nn.ReLU(),            # ReLU activation.
                nn.Linear(512, 512),  # Third fully connected layer for domain.
                nn.ReLU(),            # ReLU activation.
                nn.Linear(512, style_dim))]  # Final layer mapping to the style dimension.

    def forward(self, z, y):
        """Forward pass for the multi-domain mapping network.

        Args:
            z (torch.Tensor): Input latent vector.
            y (torch.Tensor): Domain labels (used to select the appropriate style embedding).

        Returns:
            torch.Tensor: Style embedding for the corresponding domain.
        """
        h = self.shared(z)  # Pass input through shared layers.
        out = []
        # Pass the shared representation through the unshared layers for each domain.
        for layer in self.unshared:
            out += [layer(h)]
        
        # Stack the results for each domain along a new dimension.
        out = torch.stack(out, dim=1)  # Shape: (batch_size, num_domains, style_dim)
        
        # Select the style embedding for the domain specified by y.
        idx = torch.LongTensor(range(y.size(0))).to(y.device)  # Batch index.
        s = out[idx, y]  # Select the style embedding based on the domain label y.
        return s

class MappingNetwork(nn.Module):
    """Mapping network capable of handling both single-style and multi-style embeddings 
    across multiple domains, with optional support for multimodal input."""
    def __init__(self,
                 latent_dim=160, 
                 style_dim=64,
                 num_domains=2, 
                 num_layers=4,
                 hidden_dim=512,
                 single_style=True,
                 multimodal=False, 
                 modality_list=None):
        """
        Args:
            latent_dim (int or dict): Dimensionality of latent input. Can be dict if multimodal.
            style_dim (int): Dimensionality of the style embedding.
            num_domains (int): Number of target domains for multi-style settings.
            num_layers (int): Number of layers in the network.
            hidden_dim (int): Dimensionality of hidden layers.
            single_style (bool): Flag for whether to use a single style embedding per domain.
            multimodal (bool): Flag for multimodal inputs (used in IMPA settings).
            modality_list (list, optional): List of modalities for multimodal input.
        """
        super().__init__()
        self.multimodal = multimodal
        # Define the mapping network based on configuration.
        if single_style:  # Single-style mapping (IMPA case).
            if multimodal:  # Multimodal input (different latents for each modality).
                self.mapping_network = []
                for mod in modality_list:  # Modalities can be e.g. different types of perturbations. 
                    self.mapping_network.append(
                        MappingNetworkSingleStyle(latent_dim[mod], style_dim, hidden_dim, num_layers)
                    )
                self.mapping_network = torch.nn.ModuleList(self.mapping_network)  # Convert to ModuleList.
            else:  # Single-style, single latent mapping.
                self.mapping_network = MappingNetworkSingleStyle(latent_dim, style_dim, hidden_dim, num_layers)
        else:  # Multi-style mapping (across domains).
            if multimodal:
                raise NotImplementedError  # Multi-style + multimodal not implemented.
            else:
                self.mapping_network = MappingNetworkMultiStyle(latent_dim, style_dim, num_domains)
        
        self.single_style = single_style  # Whether single-style or multi-style is used.
        
    def forward(self, z, mol=None, y=None):
        """Forward pass for the mapping network.

        Args:
            z (torch.Tensor): Latent vector input.
            mol (torch.Tensor, optional): Domain label for multi-style cases.
            y (torch.Tensor, optional): Modality index for multimodal cases.

        Returns:
            torch.Tensor: Style embedding or embeddings for the corresponding domain or modality.
        """
        if self.single_style:  # If using single-style mapping.
            if self.multimodal:
                return self.mapping_network[y](z)  # Select modality-specific mapping.
            else:
                return self.mapping_network(z)  # Single-style mapping.
        else:  # Multi-style mapping.
            if self.multimodal:
                raise NotImplementedError  # Multi-style + multimodal not implemented.
            else:
                return self.mapping_network(z, mol)  # Multi-style mapping based on domain.

################## Network aligning image space to perturbation representation ##################

class StyleEncoder(nn.Module):
    """Encoder that transforms images into style vectors."""
    def __init__(self, 
                 img_size=96,
                 style_dim=64,
                 max_conv_dim=512,
                 in_channels=3, 
                 dim_in=64, 
                 single_style=True,
                 num_domains=None):
        """
        Args:
            img_size (int): Input image size.
            style_dim (int): Dimensionality of the style vector.
            max_conv_dim (int): Maximum number of convolution filters.
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            dim_in (int): Initial number of filters.
            single_style (bool): Whether to generate a single style vector or multiple per domain.
            num_domains (int, optional): Number of target domains (only relevant if single_style=False).
        """
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(in_channels, dim_in, 3, 1, 1)]  # Initial convolution block.

        # Downsample the image, halving spatial dimensions until reaching 3x3 resolution.
        repeat_num = math.ceil(np.log2(img_size)) - 2
        final_conv_dim = img_size // (2**repeat_num)
        
        # Add residual blocks for downsampling.
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        # Final convolution to downsample spatial dimensions to 1x1.
        blocks += [nn.Conv2d(dim_out, dim_out, final_conv_dim, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.conv = torch.nn.Sequential(*blocks)  # Sequential model to hold all blocks.
        
        if not single_style:  # Multi-domain case: one style per domain.
            self.unshared = nn.ModuleList()
            for _ in range(num_domains):
                self.unshared += [nn.Linear(dim_out, style_dim)]  # Linear layers for each domain.
        else:  # Single-style case.
            self.linear = nn.Linear(dim_out, style_dim)  # Linear layer to output a single style vector.
            
        self.single_style = single_style  # Flag indicating single or multi-style.

    def forward(self, x, y=None):
        """Forward pass for the encoder.

        Args:
            x (torch.Tensor): Input images.
            y (torch.Tensor, optional): Domain labels for multi-style encoding.

        Returns:
            torch.Tensor: Style vector(s).
        """
        # Apply the shared convolution layers and flatten the output.
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        
        if not self.single_style:  # Multi-style case.
            out = []
            for layer in self.unshared:
                out += [layer(h)]
            out = torch.stack(out, dim=1)  # Stack style vectors per domain.
            idx = torch.LongTensor(range(y.size(0))).to(y.device)
            z_style = out[idx, y]  # Select style vector for the given domain.
        else:  # Single-style case.
            z_style = self.linear(h)
        return z_style

################## Network aligning image space to perturbation representation ##################

class Discriminator(nn.Module):
    """Discriminator network for the GAN model."""
    def __init__(self, 
                 img_size=96,
                 num_domains=2,
                 max_conv_dim=512, 
                 in_channels=3, 
                 dim_in=64, 
                 multi_task=True,
                 multimodal=False, 
                 modality_list=None):
        """
        Args:
            img_size (int): Size of the input images.
            num_domains (int): Number of target domains.
            max_conv_dim (int): Maximum number of convolution filters.
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            dim_in (int): Initial number of filters.
            multi_task (bool): If True, allows multi-task learning.
            multimodal (bool): If True, enables multi-modal discriminators.
            modality_list (list, optional): List of modalities for multi-modal settings.
        """
        super().__init__()
        self.multi_task = multi_task
        self.multimodal = multimodal
        blocks = []
        blocks += [nn.Conv2d(in_channels, dim_in, 3, 1, 1)]  # Initial convolution block.

        # Calculate number of downsampling steps.
        repeat_num = math.ceil(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]  # Add residual blocks for downsampling.
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2, inplace=True)]
        blocks += [nn.Conv2d(dim_out, dim_out, 3, 1, 0)]  # Final convolution block before the head.
        blocks += [nn.LeakyReLU(0.2, inplace=True)]
        
        self.conv_blocks = nn.Sequential(*blocks)  # Sequential model for convolution blocks.
        
        # Define the head based on the multimodal setting.
        if not multimodal:
            self.head = nn.Conv2d(dim_out, num_domains, 1, 1, 0)  # Single head for all domains.
        else:
            self.head = []  # One discriminator head per modality.
            for mod in modality_list:
                self.head.append(nn.Conv2d(dim_out, num_domains[mod], 1, 1, 0))
            self.head = torch.nn.ModuleList(self.head)

    def forward(self, x, mol, y):
        """Forward pass for the discriminator.

        Args:
            x (torch.Tensor): Input images.
            mol (torch.Tensor): Modality labels for multi-task learning.
            y (torch.Tensor): Domain labels for multimodal discriminators.

        Returns:
            torch.Tensor: Output of the discriminator.
        """
        # Apply convolution blocks to the input.
        out = self.conv_blocks(x)
        
        # Select the appropriate head based on multimodal setting.
        if self.multimodal:
            out = self.head[y](out)  # Use head for the specified domain.
        else:
            out = self.head(out)  # Use single head.

        out = out.view(out.size(0), -1)  # Flatten the output to (batch, num_domains).
        
        if self.multi_task:  # If multi-task learning is enabled.
            idx = torch.LongTensor(range(mol.size(0))).to(mol.device)
            out = out[idx, mol]  # Select output based on modality labels.
            
        return out  # Return the discriminator output.
    
################## Function building IMPA model ##################

def build_model(args, num_domains, device, multimodal, batch_correction, modality_list, latent_dim):
    """
    Builds the GAN model including the generator, style encoder, discriminator, and mapping network.

    Args:
        args (argparse.Namespace): Arguments containing configuration settings.
        num_domains (int): The number of domains for the task.
        device (torch.device): The device to run the model on (CPU or GPU).
        multimodal (bool): Flag indicating if the model is multimodal.
        batch_correction (bool): Flag for batch normalization correction.
        modality_list (list): List of modalities for the model.
        latent_dim (dict): Dictionary containing latent dimensions for different modalities.

    Returns:
        Munch: A dictionary-like object containing initialized models.
    """
    
    # Initialize the generator
    generator = nn.DataParallel(Generator(args.img_size,
                                          args.style_dim, 
                                          in_channels=args.n_channels, 
                                          dim_in=args.dim_in).to(device))
    
    # Initialize the style encoder
    style_encoder = nn.DataParallel(StyleEncoder(args.img_size,
                                                 args.style_dim, 
                                                 in_channels=args.n_channels, 
                                                 dim_in=args.dim_in, 
                                                 single_style=args.single_style, 
                                                 num_domains=num_domains).to(device))
    
    # Initialize the discriminator network
    discriminator = nn.DataParallel(Discriminator(args.img_size,
                                                  num_domains, 
                                                  in_channels=args.n_channels, 
                                                  dim_in=args.dim_in,
                                                  multimodal=multimodal, 
                                                  modality_list=modality_list).to(device))

    # Define input dimension based on the multimodal setting and whether single style is used
    if multimodal:
        # For multimodal settings
        if args.single_style:
            # Condition embeddings dimension
            condition_embedding_dimension = args.condition_embedding_dimension if args.use_condition_embeddings else 0
            # Total input dimensions for each modality
            input_dim = {mod: dim + condition_embedding_dimension + args.z_dimension for mod, dim in latent_dim.items()}  
        else:
            raise NotImplementedError("Multi-domain mapping for non-single style not implemented.")

    else:
        # For single domain settings
        if args.single_style:
            input_dim = args.latent_dim + args.z_dimension
        else:
            input_dim = args.z_dimension

    # Initialize the mapping network
    mapping_network = nn.DataParallel(MappingNetwork(input_dim,
                                                     args.style_dim,
                                                     num_domains=num_domains,
                                                     num_layers=args.num_layers_mapping_net, 
                                                     hidden_dim=512,
                                                     single_style=args.single_style, 
                                                     multimodal=args.multimodal, 
                                                     modality_list=modality_list).to(device))

    # Dictionary with the initialized modules
    nets = Munch(generator=generator,
                  style_encoder=style_encoder, 
                  discriminator=discriminator, 
                  mapping_network=mapping_network)
    
    return nets

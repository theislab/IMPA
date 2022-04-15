from .modules.convnet.convnet_architecture import Encoder, Decoder
from .modules.resnet.resnet_encoder import *
from .modules.resnet.resnet_decoder import *
from .modules.unet.unet import UNetEncoder, UNetDecoder 


def initialize_encoder_decoder(in_channels, in_width, in_height, variational, hparams):
    """Initialize encoder and decoder architectures 

    Args:
        hparams (dict): dictionary of hyperparameters 

    Returns:
        tuple: Encoder and decoder modules  
    """

    if hparams["autoencoder_type"] == 'resnet':
        # The different kinds of resnet
        resnet_types = {'resnet18': (resnet18, decoder18),
                        'resnet34': (resnet34, decoder34),
                        'resnet50': (resnet50, decoder50)}

        encoder = resnet_types[hparams['resnet_type']][0](in_channels = in_channels,
                latent_dim = hparams["latent_dim"],
                init_fm = hparams["init_fm"],
                in_width = in_width,
                in_height = in_height,
                variational = variational)

        decoder = resnet_types[hparams['resnet_type']][1](out_channels = in_channels,
                latent_dim = hparams["latent_dim"],
                init_fm = hparams["init_fm"],
                out_width = in_width,
                out_height = in_height,
                variational = variational)

    elif hparams["autoencoder_type"] == 'convnet':
        encoder = Encoder(
            in_channels = in_channels,
            latent_dim =  hparams["latent_dim"],
            init_fm =  hparams["init_fm"],
            n_conv =  hparams["n_conv"],
            n_residual_blocks =  hparams["n_residual_blocks"], 
            in_width =  in_width,
            in_height =  in_height,
            variational =  variational,
            batch_norm_layers_ae =  hparams["batch_norm_layers_ae"],
            dropout_ae =  hparams["dropout_ae"],
            dropout_rate_ae =  hparams["dropout_rate_ae"]
        )

        decoder = Decoder(
            out_channels =  in_channels,
            latent_dim = hparams["latent_dim"],
            init_fm =  hparams["init_fm"],
            n_conv =  hparams["n_conv"],
            n_residual_blocks =  hparams["n_residual_blocks"],  
            out_width =  in_width,
            out_height =  in_height,
            variational =  variational,
            batch_norm_layers_ae =  hparams["batch_norm_layers_ae"],
            dropout_ae =  hparams["dropout_ae"],
            dropout_rate_ae =  hparams["dropout_rate_ae"]
        ) 
    
    else:
        encoder = UNetEncoder(
            in_channels =  in_channels,
            latent_dim =  hparams["latent_dim"],
            init_fm =  hparams["init_fm"],
            n_conv =  hparams["n_conv"],
            in_width =  in_width,
            in_height =  in_height,
            variational =  variational,
        )

        decoder = UNetDecoder(
            out_channels =  in_channels,
            latent_dim = hparams["latent_dim"],
            init_fm =  hparams["init_fm"],
            n_conv =  hparams["n_conv"],
            out_width =  in_width,
            out_height =  in_height,
            variational =  variational,
        ) 
    return encoder, decoder
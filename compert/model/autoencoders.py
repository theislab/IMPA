from .modules.convnet.convnet_architecture import Encoder, Decoder
from .modules.unet.unet import UNetEncoder, UNetDecoder 
from .modules.resnet.resnet_drit.resnet_drit import ResnetDritEncoder, ResnetDritDecoder
from .modules.resnet.resnet_cyclegan.resnet_cyclegan import ResnetEncoderCycleGAN, ResnetDecoderCycleGAN



def initialize_encoder_decoder(in_channels, in_width, in_height, variational, hparams, decoding_style, extra_fm):
    """Initialize encoder and decoder architectures 

    Args:
        hparams (dict): dictionary of hyperparameters 

    Returns:
        tuple: Encoder and decoder modules  
    """

    if hparams["autoencoder_type"] == 'resnet_drit':
        encoder = ResnetDritEncoder(in_channels = in_channels,
                init_fm = hparams["init_fm"],
                n_conv = hparams["n_conv"],
                n_residual_blocks = hparams["n_residual_blocks"], 
                in_width = in_width,
                in_height = in_width,
                variational = variational)

        decoder = ResnetDritDecoder(out_channels = 3,
                init_fm = hparams["init_fm"],
                n_conv = hparams["n_conv"],
                n_residual_blocks = hparams["n_residual_blocks"], 
                out_width = in_width,
                out_height = in_width,
                decoding_style = decoding_style, 
                extra_dim = extra_fm) 


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
            decoding_style = decoding_style, 
            extra_fm = extra_fm
        ) 
    
    elif hparams["autoencoder_type"] == 'resnet_cyclegan':
        encoder = ResnetEncoderCycleGAN(in_channels = in_channels,
                init_fm = hparams["init_fm"],
                n_conv = hparams["n_conv"],
                n_residual_blocks = hparams["n_residual_blocks"], 
                in_width = in_width,
                in_height = in_height,
                variational = variational)

        decoder = ResnetDecoderCycleGAN(out_channels = in_channels,
                init_fm = hparams["init_fm"],
                n_conv = hparams["n_conv"],
                out_width = in_width,
                out_height = in_height,
                variational = variational,
                decoding_style = decoding_style, 
                extra_fm = extra_fm) 

    elif hparams["autoencoder_type"] == 'unet':
        encoder = UNetEncoder(in_channels = in_channels,
                init_fm = hparams["init_fm"],
                n_conv = hparams["n_conv"],
                in_width = in_width,
                in_height = in_height,
                variational = variational)

        decoder = UNetDecoder(out_channels = in_channels,
                init_fm = hparams["init_fm"],
                n_conv = hparams["n_conv"],
                out_width = in_width,
                out_height = in_height,
                variational = variational, 
                extra_fm = extra_fm)        

    else:
        raise NotImplementedError

    return encoder, decoder
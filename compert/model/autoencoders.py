from .modules.convnet.convnet_architecture import Encoder, Decoder
from .modules.resnet.resnet_encoder import *
from .modules.resnet.resnet_decoder import *
from .modules.unet.unet import UNetEncoder, UNetDecoder 


def initialize_encoder_decoder(self, hparams):
    """Initialize encoder and decoder architectures 

    Args:
        hparams (dict): dictionary of hyperparameters 

    Returns:
        tuple: Encoder and decoder modules  
    """
    # If the embeddings are concatenated and not summed we have to increase the input latent dimension of the decoder 
    if self.hparams["concat_embedding"]:
        # If we concatenate the drug/moa one hot --> sum the latent dim by the number of drugs or moas in the dataset 
        if self.hparams["concat_one_hot"]:
            moa_concat_dim = self.n_moa if self.predict_moa else 0  
            latent_dim_decoder = self.hparams["latent_dim"] +  self.n_seen_drugs + moa_concat_dim
        else:   
            latent_dim_decoder = self.hparams["latent_dim"] +  self.hparams["drug_embedding_dimension"] + self.hparams["moa_embedding_dimension"]
    else:
        latent_dim_decoder = self.hparams["latent_dim"]

    if hparams["autoencoder_type"] == 'resnet':
        # The different kinds of resnet
        resnet_types = {'resnet18': (resnet18, decoder18),
                        'resnet34': (resnet34, decoder34),
                        'resnet50': (resnet50, decoder50)}

        encoder = resnet_types[hparams['resnet_type']][0](in_channels = self.in_channels,
                latent_dim = self.hparams["latent_dim"],
                init_fm = self.hparams["init_fm"],
                in_width = self.in_width,
                in_height = self.in_height,
                variational = self.variational)

        decoder = resnet_types[hparams['resnet_type']][1](out_channels = self.in_channels,
                latent_dim = latent_dim_decoder,
                init_fm = self.hparams["init_fm"],
                out_width = self.in_width,
                out_height = self.in_height,
                variational = self.variational)

    elif hparams["autoencoder_type"] == 'convnet':
        encoder = Encoder(
            in_channels = self.in_channels,
            latent_dim = self.hparams["latent_dim"],
            init_fm = self.hparams["init_fm"],
            n_conv = self.hparams["n_conv"],
            n_residual_blocks = self.hparams["n_residual_blocks"], 
            in_width = self.in_width,
            in_height = self.in_height,
            variational = self.variational,
            batch_norm_layers_ae = self.hparams["batch_norm_layers_ae"],
            dropout_ae = self.hparams["dropout_ae"],
            dropout_rate_ae = self.hparams["dropout_rate_ae"]
        )

        decoder = Decoder(
            out_channels = self.in_channels,
            latent_dim = latent_dim_decoder,
            init_fm = self.hparams["init_fm"],
            n_conv = self.hparams["n_conv"],
            n_residual_blocks = self.hparams["n_residual_blocks"],  
            out_width = self.in_width,
            out_height = self.in_height,
            variational = self.variational,
            batch_norm_layers_ae = self.hparams["batch_norm_layers_ae"],
            dropout_ae = self.hparams["dropout_ae"],
            dropout_rate_ae = self.hparams["dropout_rate_ae"]
        ) 
    
    else:
        encoder = UNetEncoder(
            in_channels = self.in_channels,
            latent_dim = self.hparams["latent_dim"],
            init_fm = self.hparams["init_fm"],
            n_conv = self.hparams["n_conv"],
            n_residual_blocks = self.hparams["n_residual_blocks"], 
            in_width = self.in_width,
            in_height = self.in_height,
            variational = self.variational,
        )

        decoder = UNetDecoder(
            out_channels = self.in_channels,
            latent_dim = latent_dim_decoder,
            init_fm = self.hparams["init_fm"],
            n_conv = self.hparams["n_conv"],
            n_residual_blocks = self.hparams["n_residual_blocks"],  
            out_width = self.in_width,
            out_height = self.in_height,
            variational = self.variational,
        ) 
    return encoder, decoder
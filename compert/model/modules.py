from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F



class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """
    def __init__(
        self,
        sizes,
        batch_norm=True,
        last_layer_act="linear",
        append_layer_width=None,
        append_layer_position=None,
    ):
        super(MLP, self).__init__()
        # Setup the linear layers of the MLP 
        layers = []
        # Each layer is a 
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.ReLU(),
            ]

        layers = [l for l in layers if l is not None][:-1]
        
        # Last layer activation 
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        # Add additional layer either in first or last postion and keep the name consistent
        if append_layer_width:
            assert append_layer_position in ("first", "last")
            # Append layer in first position 
            if append_layer_position == "first":
                layers_dict = OrderedDict()
                layers_dict["append_linear"] = torch.nn.Linear(
                    append_layer_width, sizes[0]
                )
                layers_dict["append_bn1d"] = torch.nn.BatchNorm1d(sizes[0])
                layers_dict["append_relu"] = torch.nn.ReLU()
                for i, module in enumerate(layers):
                    layers_dict[str(i)] = module  # Add the previously reported layers 
            else:
                layers_dict = OrderedDict(
                    {str(i): module for i, module in enumerate(layers)}
                )
                layers_dict["append_bn1d"] = torch.nn.BatchNorm1d(sizes[-1])
                layers_dict["append_relu"] = torch.nn.ReLU()
                layers_dict["append_linear"] = torch.nn.Linear(
                    sizes[-1], append_layer_width
                )  # Append a linear layer in last position with linear activation function 
        else:
            layers_dict = OrderedDict(
                {str(i): module for i, module in enumerate(layers)}
            )

        self.network = torch.nn.Sequential(layers_dict)

    def forward(self, X):
        if self.activation == "ReLU":
            X = self.network(X)
            dim = X.size(1) // 2
            return torch.cat((self.relu(X[:, :dim]), X[:, dim:]), dim=1)
        return self.network(X)


#-------------------------------------------------------------------------------------


class GeneralizedSigmoid(torch.nn.Module):
    """
    Sigmoid, log-sigmoid or linear functions for encoding for drug perurbations.
    """

    def __init__(self, dim, device, nonlin="logsigm"):
        """Sigmoid modeling of continuous variable. It enforces f(0)=0 on the drug embedding 
        Params
        ------
        nonlin : str (default: logsigm)
            One of logsigm, sigm or None. If None, then the doser is disabled and just returns the dosage unchanged.
        """
        super(GeneralizedSigmoid, self).__init__()
        assert nonlin in ("sigm", "logsigm", None)
        self.nonlin = nonlin
        # Weights and the bias of the dose encoder  
        self.beta = torch.nn.Parameter(
            torch.ones(1, dim, device=device), requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, dim, device=device), requires_grad=True
        )

    def forward(self, X, idx=None):
        if self.nonlin == "logsigm":
            if idx is None:
                c0 = self.bias.sigmoid()
                return (torch.log1p(X) * self.beta + self.bias).sigmoid() - c0
            else:
                bias = self.bias[0][idx]
                beta = self.beta[0][idx]
                c0 = bias.sigmoid()
                return (torch.log1p(X) * beta + bias).sigmoid() - c0
        elif self.nonlin == "sigm":
            if idx is None:
                c0 = self.bias.sigmoid()
                return (X * self.beta + self.bias).sigmoid() - c0
            else:
                bias = self.bias[0][idx]
                beta = self.beta[0][idx]
                c0 = bias.sigmoid()
                return (X * beta + bias).sigmoid() - c0
        else:
            return X

    def one_drug(self, X, i):
        if self.nonlin == "logsigm":
            c0 = self.bias[0][i].sigmoid()
            return (torch.log1p(X) * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        elif self.nonlin == "sigm":
            c0 = self.bias[0][i].sigmoid()
            return (X * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        else:
            return X


def gaussian_nll(mu, log_sigma, x):
    """
    Compute the Gaussian negative log-likelihood loss
    
    mu: mean
    log_sigma: log standard deviation
    x: observation
    """
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

# Softclip
def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


#-------------------------------------------------------------------------------------

"""
Convolutional layer with residual connection 
"""

# Convolutional layer with residual connection 
class ResidualLayer(torch.nn.Module):
    """
    Simple residual block 
    """
    def __init__(self, in_channels, out_channel):
        super(ResidualLayer, self).__init__()
        # Residual unit 
        self.resblock = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channel, kernel_size = 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channel, out_channel, kernel_size = 1)
            )
        self.activation_out = torch.nn.LeakyReLU()

    def forward(self, X):
        out = self.resblock(X)
        out += X  # Residual connection 
        out = self.activation_out(out)
        return out



#-------------------------------------------------------------------------------------

"""
Encoder and Decoder classes 
"""

class Encoder(torch.nn.Module):
    def __init__(self,
                in_channels: int = 5,
                latent_dim: int = 512,
                init_fm: int = 64,
                n_conv: int = 3,
                n_residual_blocks: int = 6, 
                in_width: int = 64,
                in_height: int = 64,
                variational: bool = True, 
                batch_norm_layers_ae: bool = False,
                dropout_ae: bool = False,
                dropout_rate_ae: float = 0 ) -> None:
        super(Encoder, self).__init__() 
    
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.init_fm = init_fm  # First number of feature maps
        self.n_conv = n_conv 
        self.n_residual_blocks = n_residual_blocks
        self.in_width, self.in_height = in_width, in_height
        self.variational = variational

        # Batch norm and dropout 
        self.batch_norm_layers_ae = batch_norm_layers_ae
        self.dropout_ae = dropout_ae
        self.dropout_rate_ae = dropout_rate_ae

        # List containing the modules 
        self.modules = []

        # Build convolutional layers 
        in_fm = self.in_channels 
        out_fm = self.init_fm
        kernel_size = 3  # Initial kernel size 

        # Build downsizing convolutions 
        for i in range(0, self.n_conv):
            stride = 1 if i == 0 else 2 
            self.modules += [torch.nn.Conv2d(in_fm, out_fm,
                                kernel_size=kernel_size, 
                                stride=stride, padding=(kernel_size-1)//2)]    
                                
            if i==0 or self.batch_norm_layers_ae:
                self.modules += [torch.nn.BatchNorm2d(out_fm)]

            self.modules += [torch.nn.ReLU()]
            if self.dropout_ae:
                self.modules += [torch.nn.Dropout(p=self.dropout_rate_ae, inplace=True)]

            in_fm = out_fm
            out_fm = out_fm*2
            kernel_size += 1

        # Add residual blocks 
        for _ in range(self.n_residual_blocks):
            self.modules.append(ResidualLayer(in_fm, in_fm))

        # Append last strided convolution 
        # self.modules += [torch.nn.Conv2d(in_fm, in_fm,
        #                         kernel_size=3, 
        #                         stride=2, padding=0),
        #                         torch.nn.ReLU()] 

        self.encoder = torch.nn.Sequential(*self.modules)

        # Add bottleneck
        downsampling_factor_width = int(self.in_width//2**(self.n_conv-1))
        downsampling_factor_height = int(self.in_height//2**(self.n_conv-1))
        self.flattened_dim = in_fm*downsampling_factor_width*downsampling_factor_height
        self.flatten = torch.nn.Flatten()   

        # Can select either a variational autoencoder or a deterministic one 
        if variational:
            self.fc_mu = torch.nn.Linear(self.flattened_dim, self.latent_dim)  # Mean encoding
            self.fc_var = torch.nn.Linear(self.flattened_dim, self.latent_dim)  # Log-var encodings 
        else:
            self.fc_z = torch.nn.Linear(self.flattened_dim, self.latent_dim)

    def forward(self, X):
        X = self.encoder(X)  # Encode the image 
        X = self.flatten(X)  

        # Derive the encodings for the mean and the log variance
        if self.variational:
            mu = self.fc_mu(X)
            log_sigma = self.fc_var(X)
            return [mu, log_sigma]
        else:
            z = self.fc_z(X)
            return z


class Decoder(torch.nn.Module):
    def __init__(self,
                out_channels: int = 5,
                latent_dim: int = 512,
                init_fm: int = 64,
                n_conv: int = 3,
                n_residual_blocks: int = 6, 
                out_width: int = 64,
                out_height: int = 64,
                variational: bool = True,
                batch_norm_layers_ae: bool = False,
                dropout_ae: bool = False,
                dropout_rate_ae: float = 0) -> None:

        super(Decoder, self).__init__() 
        
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.n_conv = n_conv 
        self.init_fm = init_fm*(2**(self.n_conv-1))  # The first number of feature vectors 
        self.n_residual_blocks = n_residual_blocks
        self.out_width, self.out_height = out_width, out_height
        
        # Build convolutional dimensions
        self.modules = []

        # Layer to upscale latent sample 
        self.upsampling_factor_width = self.out_width//2**(self.n_conv-1)
        self.upsampling_factor_height = self.out_height//2**(self.n_conv-1)
        self.flattened_dim = self.init_fm*self.upsampling_factor_width*self.upsampling_factor_height
        self.upsample_fc = torch.nn.Linear(self.latent_dim, self.flattened_dim)
        self.variational = variational

        # Batch norm and dropout 
        self.batch_norm_layers_ae = batch_norm_layers_ae
        self.dropout_ae = dropout_ae
        self.dropout_rate_ae = dropout_rate_ae

        in_fm = self.init_fm
        out_fm = self.init_fm//2
        kernel_size = 6  # Initial kernel size 

        # First upsampling layer
        # self.modules += [torch.nn.ConvTranspose2d(in_fm, in_fm,
        #                         kernel_size=3, 
        #                         stride=2, padding=0),
        #                         torch.nn.ReLU()]

        # Append the residual blocks
        for _ in range(self.n_residual_blocks):
            self.modules.append(ResidualLayer(in_fm, in_fm))

        for i in range(0, self.n_conv):
            stride = 2 if i < self.n_conv-1 else 1
            self.modules += [torch.nn.ConvTranspose2d(in_fm, out_fm,
                                kernel_size=kernel_size, stride=stride, padding=2)]
            if self.batch_norm_layers_ae:   
                self.modules += [torch.nn.BatchNorm2d(out_fm)]

            if i > 0:
                kernel_size -= 1
            
            self.modules += [torch.nn.ReLU() if i<self.n_conv-1 else torch.nn.Sigmoid()]
            if self.dropout_ae and i < self.n_conv-1:
                self.modules += [torch.nn.Dropout(p=self.dropout_rate_ae, inplace=True)]

            in_fm = out_fm
            if i == self.n_conv-2:
                out_fm = self.out_channels
            else:
                out_fm = out_fm//2 
            
            # if i < self.n_conv-1:
            #     for _ in range(self.n_residual_blocks):
            #         self.modules.append(ResidualLayer(in_fm, in_fm))

        # Assemble the decoder
        self.decoder = torch.nn.Sequential(*self.modules)
    
    def forward(self, z):
        X = self.upsample_fc(z)
        # Reshape to height x width
        X = X.view(-1, self.init_fm, self.upsampling_factor_width, self.upsampling_factor_height)
        X = self.decoder(X)
        return X 
        
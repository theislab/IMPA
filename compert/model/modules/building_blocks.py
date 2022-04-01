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
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=F.relu, dropout=0.0):
        """
        Generalized MLP model.

        Args:
            input_dim (int): Number of input features.
            hidden_dims (list of int): List containing the number of neurons in each hidden layer.
            output_dim (int): Number of output neurons.
            activation (callable): Activation function to use (default: ReLU).
            dropout (float): Dropout probability (default: 0.0).
        """
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        return self.output_layer(x)


def count_parameters(param_list):
    """Counts the total number of elements in a list of parameters."""
    return sum(param.numel() for _, param in param_list)


class HyperNet(nn.Module):
    """
    A fully connected network that outputs a flat vector of the required number of weights.
    """

    def __init__(self, primary_net, latent_dim, hidden_dims):
        super().__init__()
        self.primary_net = primary_net
        self.primary_params = sum(p.numel() for p in self.primary_net.parameters())

        # Freeze primary network weights
        for param in primary_net.parameters():
            param.requires_grad = False

        self.latent_dim = latent_dim
        self.hyper_net = MLP(latent_dim, hidden_dims, self.primary_params)

    def forward(self, z):
        return self.hyper_net(z)

    def sample_network(self, device):
        z = torch.randn(self.latent_dim, device=device)
        weights = self.hyper_net(z)

        i = 0
        weight_dict = {}
        for k, v in self.primary_net.state_dict().items():
            weight_dict[k] = weights[i : i + v.numel()].view(v.shape)
            i += v.numel()

        return partial(torch.func.functional_call, self.primary_net, weight_dict)

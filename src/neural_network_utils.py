import torch
import torch.nn as nn


def init_weights_linear_layer(l):
    """Customised Gaussian initialisation for torch.nn.Linear layers"""
    if isinstance(l, nn.Linear):
        mean = 0.0
        std = 0.01
        torch.nn.init.normal_(l.weight, mean=mean, std=std)
        torch.nn.init.zeros_(l.bias)


def simple_nn(input_size, output_size, hidden_units=500):
    """Simple neural network used for overfitting experiments"""
    layers = nn.ModuleList()
    layers.append(nn.Linear(in_features=input_size, out_features=hidden_units))
    layers.append(nn.LeakyReLU())
    layers.append(nn.Linear(in_features=hidden_units, out_features=output_size))

    return layers


def small_nn(input_size=2, output_size=2, hidden_units=50):
    """Small neural network used for toy example"""
    layers = nn.ModuleList()

    layers.append(nn.Linear(in_features=input_size, out_features=hidden_units))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(in_features=hidden_units, out_features=output_size))
    layers.apply(init_weights_linear_layer)

    return layers


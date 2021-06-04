import torch
import torch.nn as nn
import torch.optim as torch_optim

from src.models.discriminative_parent_model import DiscriminativeParentModel


class DiscriminativeModelMAE(DiscriminativeParentModel):
    """ Standard classifier neural network, MAE loss
    """

    def __init__(self, model_layers, num_classes=10, learning_rate=0.001, loss=nn.L1Loss(),
                 device=torch.device("cpu")):

        super().__init__(loss=loss, device=device)

        self.layers = model_layers.to(self.device)
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.optimizer = torch_optim.Adam(self.parameters(), lr=self.learning_rate)

        self.name = "discriminative_model_mae"

    def _forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

    def _output_transform(self, output):
        return nn.Softmax(dim=1)(output)

    def _predict_transform(self, output):
        return nn.Softmax(dim=1)(output)

    def _modify_targets(self, targets, inputs):
        return targets








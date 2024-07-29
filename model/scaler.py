import torch
from torch import nn


class Scaler(nn.Module):
    def __init__(self, input_dim, tau=0.5):
        super(Scaler, self).__init__()
        self.tau = tau
        self.scale = nn.Parameter(torch.zeros(input_dim))

    def forward(self, inputs, mode="positive"):
        if mode == "positive":
            scale = self.tau + (1 - self.tau) * torch.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * torch.sigmoid(-self.scale)
        return inputs * torch.sqrt(scale)

import torch
from torchvision import models
from torch import nn

class AgeGenderModel(nn.Module):
    def __init__(self, n_bins: int, n_binsets: int):
        self.vgg = models.vgg16()
        self.n_bins = n_bins
        self.bin_layers = [nn.Linear(n_bins, n_bins, bias=False) for _ in range(n_binsets)]

    def forward(self, x):
        pass
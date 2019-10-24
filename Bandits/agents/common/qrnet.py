import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def init_weights(m, mean, std):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean, std)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, mean, std)

class QRNet(nn.Module):
    """
    The quantile estimator modelized by a NN.
    Initialized with respect to a normal prior (mean_prior, std_prior)
    """

    def __init__(self, n_features, num_quants, mean_prior, std_prior):
        super(QRNet, self).__init__()
        self.n_features = n_features
        self.num_quants  = num_quants
        self.name = "QRDQN"
        
        self.features = nn.Sequential(
            nn.Linear(self.n_features, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.num_quants))

        self.features.apply(lambda m: init_weights(m, mean_prior, std_prior))

    def forward(self, x):
        batch_size = x.size(0)
        out = self.features(x)
        out = out.view(batch_size, self.num_quants)
        return out

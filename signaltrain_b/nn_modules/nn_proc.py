# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, T=25, R=2):
        super(AutoEncoder, self).__init__()

        # Parameters
        self._T = T
        self._R = R

        # Analysis
        self.fnn_enc = nn.Linear(self._T, self._R, bias=True)
        self.fnn_dec = nn.Linear(self._R, self._T, bias=True)

        # Activation functions
        self.relu = nn.ReLU()

        self.initialize()

    def initialize(self):
        torch.nn.init.xavier_normal(self.fnn_enc.weight)
        torch.nn.init.xavier_normal(self.fnn_dec.weight)
        self.fnn_enc.bias.data.zero_()
        self.fnn_dec.bias.data.zero_()

    def forward(self, x_input, skip_connections='res'):
        x_input = x_input.transpose(2, 1)

        z = self.relu(self.fnn_enc(x_input))

        if skip_connections == 'exp':           # Refering to an old AES paper for exponentiation
            z_a = torch.log(self.relu(self.fnn_dec(z)) + 1e-6) * torch.log(x_input + 1e-6)
            out = torch.exp(z_a)
        elif skip_connections == 'res':         # Refering to residual connections
            out = self.relu(self.fnn_dec(z) + x_input)
        elif skip_connections == 'sf':          # Skip-filter
            out = self.relu(self.fnn_dec(z)) * x_input
        else:
            out = self.relu(self.fnn_dec(z))

        return out.transpose(2, 1)

# EOF

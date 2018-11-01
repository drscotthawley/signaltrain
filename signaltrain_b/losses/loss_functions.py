# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import torch


def mse(x, x_hat):
    return torch.norm(torch.pow(x - x_hat, 2.))


def mae(x, x_hat):
    return torch.norm(x - x_hat, 1)

# EOF

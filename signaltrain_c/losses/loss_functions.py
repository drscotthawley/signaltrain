# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import torch


def mse(x, x_hat):
    return torch.norm(torch.pow(x - x_hat, 2.))


def mae(x, x_hat):
    return torch.norm(x - x_hat, 1)

# Alternatives to mae (see https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0)
def logcosh(x, x_hat):
    return torch.sum( torch.log( torch.cosh(x - x_hat) ))

def smoothl1(x, x_hat, delta=0.5):  # Huber loss
    return np.sum ( np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), \
        delta*np.abs(true - pred) - 0.5*(delta**2)) )

# EOF

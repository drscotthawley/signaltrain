# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import torch


#import torch.nn.functional as F
def calc_loss(x_hat,y_cuda,mag,objective,batch_size=20):
        # Reconstruction term plus regularization -> Slightly less wiggly waveform
        loss = objective(x_hat, y_cuda) + 1e-5*mag.norm(1) #- 1e-3*F.conv1d(x_hat.unsqueeze(0),y_cuda.unsqueeze(0)).norm(1) # add negative correlation loss
        return loss/batch_size


def mse(x, x_hat):
    return torch.norm(torch.pow(x - x_hat, 2.))


def mae(x, x_hat):
    return torch.norm(x - x_hat, 1)

# Alternatives to mae (see https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0)
def logcosh(x, x_hat):
    return torch.sum( torch.log( torch.cosh(x - x_hat) ))

def smoothl1(x, x_hat, delta=0.5):  # Huber loss
    #return torch.sum ( torch.where(torch.abs(true-pred) < delta , 0.5*((true-pred)**2), \
    #    delta*toch.abs(true - pred) - 0.5*(delta**2)) )
    return torch.nn.SmoothL1Loss(true-pred)
# EOF

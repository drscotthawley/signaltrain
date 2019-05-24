# -*- coding: utf-8 -*-
__author__ = 'S.H. Hawley'

"""
These routines replicate functionality in FastAI library, i.e. the 1cycle LR schedule and the LR finder.
NOTE: moved LR finder to signaltrain/utils
"""

# imports
import numpy as np
import torch


def get_1cycle_schedule(lr_max = 1e-3, n_data_points = 8000, epochs = 200,
                        batch_size = 40):
  """
  Creates a look-up table of learning rates for 1cycle schedule with cosine annealing

  Keywork inputs:
    lr_max            chosen by user after lr_finder
    n_data_points     data points per epoch (e.g. size of training set)
    epochs            number of epochs
    batch_size        batch size

  Output:
    lrs               look-up table of LR's, with length equal to total # of iterations

  Then you can use this in your PyTorch code by counting iteration number and setting
          optimizer.param_groups[0]['lr'] = lrs[iter_count]
  """
  #pct_start, div_factor = 0.3, 25.        # @sgugger's parameters in fastai code
  pct_start, div_factor = 0.3, 15.       # my attempt to train faster
  lr_start = lr_max/div_factor
  lr_end = lr_start/1e4
  n_iter = n_data_points * epochs // batch_size     # number of iterations
  a1 = int(n_iter * pct_start)                      # boundary between increasing and decreasing
  a2 = n_iter - a1

  # make look-up table
  #lrs_first = np.linspace(lr_start, lr_max, a1)            # linear growth
  lrs_first = (lr_max-lr_start)*(1-np.cos(np.linspace(0,np.pi,a1)))/2 + lr_start  # cosine growth
  lrs_second = (lr_max-lr_end)*(1+np.cos(np.linspace(0,np.pi,a2)))/2 + lr_end  # cosine annealing
  lrs = np.concatenate((lrs_first, lrs_second))

  # also schedule the momentum
  mom_min, mom_max = 0.85, 0.95
  mom_avg, mom_amp = (mom_min+mom_max)/2, (mom_max-mom_min)/2
  mom_first = mom_avg + mom_amp*np.cos(np.linspace(0,np.pi,a1))
  mom_second = mom_avg - mom_amp*np.cos(np.linspace(0,np.pi,a2))
  moms = np.concatenate((mom_first, mom_second))

  return lrs, moms

# EOF

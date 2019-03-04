# -*- coding: utf-8 -*-
__author__ = 'S.H. Hawley'

"""
These routines replicate functionality in FastAI library, i.e. the 1cycle LR schedule and the LR finder.
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
  pct_start, div_factor = 0.3, 15.        # my modification
  lr_start = lr_max/div_factor
  lr_end = lr_start/1e4
  n_iter = n_data_points * epochs // batch_size     # number of iterations
  a1 = int(n_iter * pct_start)
  a2 = n_iter - a1

  # make look-up table
  lrs_first = np.linspace(lr_start, lr_max, a1)            # linear growth
  lrs_second = (lr_max-lr_end)*(1+np.cos(np.linspace(0,np.pi,a2)))/2 + lr_end  # cosine annealing
  lrs = np.concatenate((lrs_first, lrs_second))
  return lrs



def lrfind(model, datagen, optimizer, calc_loss, start=1e-6, stop=1e-2, num_lrs=150):
    """ Learning Rate finder.  See leslie howard, sylvian gugger & jeremy howard's work """
    print("Running LR Find:",end="",flush=True)

    lrs, losses = [], []
    for lr_try in np.logspace(np.log10(start), np.log10(stop), num_lrs):
        print(".",sep="",end="",flush=True)
        optimizer.param_groups[0]['lr'] = lr_try
        x_cuda, y_cuda, knobs_cuda = datagen.new()
        x_hat, mag, mag_hat = model.forward(x_cuda, knobs_cuda)
        loss = calc_loss(x_hat,y_cuda,mag)
        lrs.append(lr_try)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        model.clip_grad_norm_()
        optimizer.step()
    plt.figure(1)
    plt.semilogx(lrs,losses)
    outfile = 'lrfind.png'
    plt.savefig(outfile)
    plt.close(plt.gcf())
    print("\nLR Find finished. See "+outfile)



if __name__ == "__main__":
    # RUN THE LR FINDER -- must be run from within directory where this source code exists

    import audio
    import sys, os
    sys.path.append('..')       # not something you want for generic import of this code, i.e. not up at the top of this code
    from nn_modules import nn_proc
    from losses import loss_functions
    import matplotlib.pylab as plt

    np.random.seed(218)
    torch.manual_seed(218)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(218)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type('torch.FloatTensor')

    # Data settings
    shrink_factor = 2  # reduce dimensionality of run by this factor
    time_series_length = 8192 // shrink_factor
    sampling_freq = 44100. // shrink_factor

    batch_size = 40

    # Effect settings
    effect=audio.Compressor_4c()

    # Analysis parameters
    ft_size = 1024 // shrink_factor
    hop_size = 384 // shrink_factor
    expected_time_frames = int(np.ceil(time_series_length/float(hop_size)) + np.ceil(ft_size/float(hop_size)))

    # Initialize nn modules
    model = nn_proc.MPAEC(expected_time_frames, ft_size=ft_size, hop_size=hop_size, n_knobs=len(effect.knob_names))
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3, weight_decay=0)

    datagen = audio.AudioDataGenerator(time_series_length, sampling_freq, effect, batch_size=batch_size, device=device)

    lrfind(model, datagen, optimizer, loss_functions.calc_loss)
# EOF

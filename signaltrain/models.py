# -*- coding: utf-8 -*-
__author__ = 'S.H. Hawley'

# imports
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import sys
from . import transforms as front_end
from . import utils as st_utils

# next 4 lines are only for modelviz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def nearest_even(x):
    """hack for making sure grid sizes can split data evenly between 2 GPUs"""
    y = int(x)
    if (0 == y % 2):
        return y
    else:
        return y+1


class SpecShrinkGrow_catri_skipadd(nn.Module):
    """  *******  'The Model' *******

    Spectral bottleneck encoder, cat real & imag parts together, skip connections are additions

    SHH: This version was responsible for a June 22, 2018 milestone of learning the compressor for random event onsets

    This model first projects the input audio to smaller sizes, using some 'dense' layers,
    Then calls the FNN analysis routine, does show shinking & growing, calls FNNSynthesis,
    and then expands back out to the original audio size.
    Two skip connections are present; more or fewer produced worse results that this.
    For a diagram of thi smodel, see ../docs/model_diagram.png
    """

    def __init__(self, chunk_size=1024):
        super(SpecShrinkGrow_catri_skipadd, self).__init__()

        ratio = 3   # Reduction ratio. 3 works about as well as 2.  4 seems to be too much
                    # one problem with ratio of 3 is it can produce odd sizes that don't split the data evenly between two GPUs
        self.mid_size = nearest_even(chunk_size/ratio)
        ft_size = nearest_even(self.mid_size /ratio)  # previously used ft_size = chunk_size, but that was limiting GPU memory usage

        # these shrink & grow routines are just to try to decrease encoder-decoder GPU memory usage
        self.front_shrink = nn.Linear(chunk_size, self.mid_size, bias=False)
        self.front_shrink2 = nn.Linear(self.mid_size, ft_size, bias=False)
        self.back_grow2    = nn.Linear(ft_size, self.mid_size, bias=False)
        self.back_grow    = nn.Linear(self.mid_size, chunk_size, bias=False)

        # the "FFT" routines
        self.encoder = front_end.FNNAnalysis(ft_size=ft_size)   # gives matrix mult. size mismatches
        self.decoder = front_end.FNNSynthesis(ft_size=ft_size)#, random_init=True)  #  random_init=True gives me better Val scores

        ft_out_dim  = int(ft_size/2+1)   # this is the size of the output from the FNNAnalysis routine
        self.full_dim = 2*ft_out_dim          # we will cat the real and imag halves together

        # define some more size variables for shrinking & growing sizes in between encoder & decoder
        self.med_dim   = nearest_even(self.full_dim / ratio)
        small_dim = nearest_even(self.med_dim / ratio)


        self.shrink  = nn.Linear(self.full_dim,  self.med_dim, bias=False)
        self.shrink2 = nn.Linear(self.med_dim,   small_dim, bias=False)
        self.grow2   = nn.Linear(small_dim, self.med_dim, bias=False)
        self.grow    = nn.Linear(self.med_dim,   self.full_dim, bias=False)
        self.act     = nn.LeakyReLU()   # Tried ReLU, ELU, SELU; Leaky seems to work best.  SELU yields instability

    def forward(self, input_var, skips=(2,3,4)):    # my trials indicate skips=(2,3) works best. (1,2,3) has more noise, (3) converges slower, no skips converges slowest
        y_s = self.act( self.front_shrink(input_var) )     # _s to prepare for skip connection "skips=3"
        y = self.act( self.front_shrink2(y_s) )
        real, imag = self.encoder(y)
        ricat_s   = torch.cat((real, imag), 2)           # 'cat' real & imag together; _s saves value for a skip later
        ricat  = self.act( self.shrink(ricat_s) )        # _s2 for other skip connection  skips=1
        ### L1 regularization goes here
        reg_term = torch.mean(torch.norm(ricat, 1, dim=-1))
        ricat  = self.act( self.shrink2(ricat) )

        # ----- here is the 'middle of the hourglass';  from here we expand

        ricat     = self.act( self.grow2(ricat) )

        ricat     = self.act( self.grow(ricat) )
        if (2 in skips):
            ricat += ricat_s   # "skip filter" connection: multiplicative rather than additive

        # split the cat-ed real & imag back up [arbitrarily] ;
        uncat = torch.chunk( ricat, 2, dim=2)

        y = self.decoder(uncat[0],uncat[1])
        y = self.act(self.back_grow2(y))
        if (3 in skips):
            y += y_s
        y = self.back_grow(y)
        if (4 in skips):  # for final skip we simply add (WaveNet does this). cat is too memory-intensive for large audio clips
            y *= input_var # Let us try it at the output
            #y += input_var

        layers = []# (ricat)     # additional container to report intermediary activations & more
        return y, layers, reg_term




def model_viz(model, outfileprefix):
    """
    Utility for visualizing aspects of the model's internal state
    Outputs to PDF file
    """
    outfile = outfileprefix+'_modelviz.pdf'
    n_modules = 0
    for module in model.modules():
        n_modules += 1

    fignum = 1
    with PdfPages(outfile) as pdf:
        for m in model.modules():
            if isinstance(m, nn.Linear):   # linear layers
                tensor = m.weight.data.cpu().numpy()
                mdescr = str(m)
                #print("Linear module: tensor.shape = ",tensor.shape)
                fig, ax1 = plt.subplots(1)
                plt.title(mdescr)
                ax1.imshow(tensor.squeeze())
                ax1.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
                pdf.savefig()
                plt.close()
                fignum += 1

            if isinstance(m, nn.Conv2d):   # conv2d layers
                tensor = m.weight.data.cpu().numpy()
                mdescr = str(m)
                # Using https://discuss.pytorch.org/t/understanding-deep-network-visualize-weights/2060/7
                #print("Conv2D or Linear module: tensor.shape = ",tensor.shape)
                num_batches = tensor.shape[0]
                num_cols = 4 #tensor.shape[1]
                num_rows = 1 #+ num_batches // num_cols
                fig = plt.figure(figsize=(num_cols,num_rows*2))
                plt.title(mdescr)
                for i in range(num_batches):
                    ax1 = fig.add_subplot(num_rows,num_cols,i+1)
                    ax1.imshow(tensor[i,0])
                    ax1.axis('off')
                    ax1.set_xticklabels([])
                    ax1.set_yticklabels([])
                plt.subplots_adjust(wspace=0.1, hspace=0.1)
                pdf.savefig()
                plt.close()
                fignum += 1

            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                tensor = m.weight.data.cpu().numpy()
                mdescr = str(m)
                # Using https://discuss.pytorch.org/t/understanding-deep-network-visualize-weights/2060/7
                #print("Conv1d module: tensor.shape = ",tensor.shape)
                fig, ax1 = plt.subplots(1)
                plt.title(mdescr)
                ax1.imshow(tensor.squeeze())
                ax1.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
                pdf.savefig()
                plt.close()
                fignum += 1
    return
#  EOF

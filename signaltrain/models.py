
__author__ = 'S.H. Hawley'

# imports
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import sys
from . import transforms as front_end

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class CNNAuto(nn.Module):   # 'convolutional autoencoder'
    def __init__(self, finaldim):
        super(CNNAuto, self).__init__()
        self.io_channels = 1   # mono for now
        self.feature_channels = 4  # chose this arbitrarily. nothing sacred about this number!
        self.kernel_size = 7
        self.pad = int(self.kernel_size/2)
        self.scale_factor = 2
        self.layer_in = nn.Sequential(     # generates feature_channels times more data
            nn.Conv2d(self.io_channels, self.feature_channels, kernel_size=self.kernel_size, padding=self.pad),
            nn.BatchNorm2d(self.feature_channels),
            nn.LeakyReLU() )

        self.layer_mid = nn.Sequential(    # keeps number of feature channels the same
            nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=self.kernel_size, padding=self.pad),
            nn.BatchNorm2d(self.feature_channels),
            nn.LeakyReLU() )

        # Pooling and Un-Pooling
        self.layer_pool = nn.MaxPool2d(self.scale_factor, padding=(1,0), return_indices=True)
        self.layer_unpool = nn.MaxUnpool2d(self.scale_factor, padding=(1,0) )

        self.layer_dropout = nn.Dropout2d()

        self.layer_out = nn.Sequential(     # combines feature channels back to original # of channels
            nn.Conv2d(self.feature_channels, self.io_channels, kernel_size=self.kernel_size, padding=self.pad),
            nn.BatchNorm2d(self.io_channels),
            nn.LeakyReLU() )

    def forward(self, x):
        #print("input           x.size() = ",x.size())

        y = self.layer_in(x)
        save_size = y.size()
        #print("after 1st conv  y.size() = ",y.size())
        '''  pooling & growing is causing problems; disabled for now '''
        y, indices = self.layer_pool(y)   #   \      /
        #y = self.layer_mid(y)
        #y = self.layer_pool(y)           #    \   /
        #y = self.layer_mid(y)
        #y = self.layer_dropout(y)
        y = self.layer_mid(y)
        #y = self.layer_dropout(y)
        y = self.layer_mid(y)
        y = self.layer_unpool(y, indices, output_size=save_size)      #    /   \
        #print("after unpool    y.size() = ",y.size())

        #y = self.layer_mid(y)
        #y = self.layer_grow2(y)            #  /      \
        y = self.layer_out(y)
        #print("after last conv y.size() = ",y.size())

        return y

# TODO: Doesn't improve results.  Unused.
class LowPassLayer(nn.Module):
    '''
    Description:
    This is a tunable 'reverse sigmoid' soft mask, which will get multiplied (element-wise)
    by the (vertical columns of) frequency spectrum, to serve as a low-pass filter.
    It has two parameters: the central frequency (from 0...1, scales with size of input)
    and a steepness parameter (-1.0=flat and 1.0=vertical)

    Purpose:
    Much of the loss-reduction in the final network output gets bound up in damping
    high-frequency noise.  In theory we could let the network do
    this eventually, but this layer is to help converge faster.

    Notes:
    Discussion & demo at ../docs/LowPassLayer.ipynb
    Layer-writing guide at https://discuss.pytorch.org/t/how-to-define-a-new-layer-with-autograd/351
    '''
    def __init__(self, bins):
        super(LowPassLayer, self).__init__()
        self.eps = 0.1
        self.bins = bins
        self.m = nn.Parameter(torch.zeros(1)).cuda()      # steepness, tunable parameter  # .cuda() actually freezes the value
        self.b = nn.Parameter(torch.zeros(1)).cuda()     # center, tunable parameter
        if torch.has_cudnn:
            self.x = Variable(torch.linspace(0, bins-1, steps=bins)).cuda()
        else:
            self.x = Variable(torch.linspace(0, bins-1, steps=bins))

    def forward(self, in_stft):
        mval, bval = self.m.data.cpu().numpy()[0],  self.b.data.cpu().numpy()[0]
        print(' m={:10.5e}'.format(mval),' b={:10.5e}'.format(bval),sep="",end="")
        m = self.m.expand_as(self.x)     # make it a 1-D vector of repeated numbers
        b = self.b.expand_as(self.x)
        mask = 1.0/(1+torch.exp( (1+m)/(1-m+self.eps) *10* (self.x /self.bins-(b+0.5)) )  )
        return in_stft * mask          # elementwise multiplication


# Just including here as an example. Unused.
# From https://discuss.pytorch.org/t/how-to-define-a-new-layer-with-autograd/351/3
class Gaussian(nn.Module):
    def __init__(self):
        super(Gaussian, self).__init__()
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.c = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # unfortunately we don't have automatic broadcasting yet
        a = self.a.expand_as(x)
        b = self.b.expand_as(x)
        c = self.c.expand_as(x)
        return a * torch.exp((x - b)^2 / c)


# 'spectral encoder-decoder'
class SpecEncDec(nn.Module):
    def __init__(self, ft_size=1024):
        super(SpecEncDec, self).__init__()
        self.encoder = front_end.FNNAnalysis(ft_size=ft_size)   # gives matrix mult. size mismatches
        self.decoder = front_end.FNNSynthesis(ft_size=ft_size)

        self.full_dim = int(ft_size/2 +1)
        self.dense = nn.Linear(self.full_dim, self.full_dim)

        self.act = nn.LeakyReLU()   # Tried ReLU, ELU, SELU; Leaky works best.  SELU yields instability

    def forward(self, input_var):
        y = input_var.unsqueeze(0)   # hack to deal with batch size of 1 right now; TODO: remove this
        real, imag = self.encoder(y)
        real, imag = self.act( self.dense(real) ), self.act( self.dense(imag) )
        real, imag = self.act( self.dense(real) ), self.act( self.dense(imag) )
        real, imag = self.act( self.dense(real) ), self.act( self.dense(imag) )
        real, imag = self.act( self.dense(real) ), self.act( self.dense(imag) )
        y = self.decoder(real, imag)
        y = y.squeeze(0)
        return y

# little hack for making sure grid sizes can split data evenly between 2 GPUs
def nearest_even(x):
    y = int(x)
    if (0 == y % 2):
        return y
    else:
        return y+1

class SpecShrinkGrow_cat(nn.Module):  # spectral bottleneck encoder
    # This model first projects the input audio to smaller sizes, using some 'dense' layers,
    #  Then calls the FNN analysis routine, does show shinking & growing, calls FNNSynthesis,
    #  and then expands back out to the original audio size.
    #  Two skip connections are present; more or fewer produced worse results that this.
    #  For a diagram of thi smodel, see ../docs/model_diagram.png
    #
    def __init__(self, chunk_size=1024):
        super(SpecShrinkGrow, self).__init__()


        ratio = 3   # Reduction ratio. 3 works about as well as 2.  4 seems to be too much
                    # one problem with ratio of 3 is it can produce odd sizes that don't split the data evenly between two GPUs
        mid_size = nearest_even(chunk_size/ratio)
        ft_size = nearest_even(mid_size /ratio)  # previously used ft_size = chunk_size, but that was limiting GPU memory usage

        # these shrink & grow routines are just to try to decrease encoder-decoder GPU memory usage
        self.front_shrink = nn.Linear(chunk_size, mid_size, bias=False)
        self.front_shrink2 = nn.Linear(mid_size, ft_size, bias=False)
        self.back_grow2    = nn.Linear(ft_size, mid_size, bias=False)
        self.back_grow    = nn.Linear(mid_size, chunk_size, bias=False)

        # the "FFT" routines
        self.encoder = front_end.FNNAnalysis(ft_size=ft_size)   # gives matrix mult. size mismatches
        self.decoder = front_end.FNNSynthesis(ft_size=ft_size)#, random_init=True)  #  random_init=True gives me better Val scores

        ft_out_dim  = int(ft_size/2+1)   # this is the size of the output from the FNNAnalysis routine
        full_dim = 2*ft_out_dim          # we will cat the real and imag halves together
        #self.cat_ri  = nn.Linear(2*full_dim,  full_dim, bias=False)

        # define some more size variables for shrinking & growing sizes in between encoder & decoder
        med_dim   = nearest_even(full_dim / ratio)
        small_dim = nearest_even(med_dim / ratio)

        #print("chunk_size, mid_size, ft_size = ",chunk_size, mid_size, ft_size)
        #print("full_dim, med_dim, small_dim = ",full_dim, med_dim, small_dim)

        self.shrink  = nn.Linear(full_dim,  med_dim, bias=False)
        self.shrink2 = nn.Linear(med_dim,   small_dim, bias=False)
        #self.dense = nn.Linear(self.small_dim, self.small_dim, bias=False)  # not needed
        self.grow2   = nn.Linear(small_dim, med_dim, bias=False)
        self.grow    = nn.Linear(med_dim,   full_dim, bias=False)

        self.mapskip  = nn.Linear(2*full_dim,  full_dim, bias=False)  # maps concatenated skip connection to 'regular' size
        self.mapskip2 = nn.Linear(2*med_dim,   med_dim, bias=False)
        self.bigskip = nn.Linear(2*mid_size,   mid_size, bias=False)
        #self.finalskip = nn.Linear(2*chunk_size,   chunk_size, bias=False)  # does not fit in CUDA memory

        self.act     = nn.LeakyReLU()   # Tried ReLU, ELU, SELU; Leaky seems to work best.  SELU yields instability

    def forward(self, input_var, skips=(2,3,4)):    # my trials indicate skips=(2,3) works best. (1,2,3) has more noise, (3) converges slower, no skips converges slowest
        y_orig = input_var
        y_s = self.act( self.front_shrink(y_orig) )     # _s to prepare for skip connection "skips=3"
        y = self.act( self.front_shrink2(y_s) )
        y = y.unsqueeze(0)   # hack to deal with batch size of 1 right now; TODO: remove this
        real, imag = self.encoder(y)
        #print("real.size() = ",real.size())
        ricat_s = torch.cat((real, imag), 2)                # 'cat' real & imag together; _s for a skip_later
        #print("ricat_s.size() = ",ricat_s.size())
        ricat_s2  = self.act( self.shrink(ricat_s) )        # _s2 for other skip connection  skips=1

        ricat = self.act( self.shrink2(ricat_s2) )


        # ----- here is the 'middle of the hourglass';  from here we expand


        ricat = self.act( self.grow2(ricat) )

        if (1 in skips):  # this one has no discernable effect; but does yield more noise
            ricat = self.mapskip2(torch.cat((ricat, ricat_s2), 2)) # make skip connection

        ricat = self.act( self.grow(ricat) )

        if (2 in skips):
            ricat = self.mapskip(torch.cat((ricat, ricat_s), 2))   # make skip connection

        uncat = torch.chunk( ricat, 2, dim=2)       # split the cat-ed real & imag back up [arbitrarily] ;
        # TODO: actually, this probably sets imag to the value of the skip connection we just made
        #print("len(uncat) = ",len(uncat))
        real, imag = uncat[0],uncat[1]

        y = self.decoder(real, imag)
        y = y.squeeze(0)

        y = self.act(self.back_grow2(y))
        if (3 in skips):
            y = self.bigskip( torch.cat((y, y_s), 1) )
        y = self.back_grow(y)
        #if (4 in skips):
        #    y = self.finalskip( torch.cat((y, y_orig), 1) )  # too memory-intensive for large audio clips
        return y


class SpecShrinkGrow(nn.Module):  # spectral bottleneck encoder
    # This model first projects the input audio to smaller sizes, using some 'dense' layers,
    #  Then calls the FNN analysis routine, does show shinking & growing, calls FNNSynthesis,
    #  and then expands back out to the original audio size.
    #  Two skip connections are present; more or fewer produced worse results that this.
    #  For a diagram of thi smodel, see ../docs/model_diagram.png
    #
    def __init__(self, chunk_size=1024):
        super(SpecShrinkGrow, self).__init__()


        ratio = 3   # Reduction ratio. 3 works about as well as 2.  4 seems to be too much
                    # one problem with ratio of 3 is it can produce odd sizes that don't split the data evenly between two GPUs
        mid_size = nearest_even(chunk_size/ratio)
        ft_size = nearest_even(mid_size /ratio)  # previously used ft_size = chunk_size, but that was limiting GPU memory usage

        # these shrink & grow routines are just to try to decrease encoder-decoder GPU memory usage
        self.front_shrink = nn.Linear(chunk_size, mid_size, bias=False)
        self.front_shrink2 = nn.Linear(mid_size, ft_size, bias=False)
        self.back_grow2    = nn.Linear(ft_size, mid_size, bias=False)
        self.back_grow    = nn.Linear(mid_size, chunk_size, bias=False)

        # the "FFT" routines
        self.encoder = front_end.FNNAnalysis(ft_size=ft_size)   # gives matrix mult. size mismatches
        self.decoder = front_end.FNNSynthesis(ft_size=ft_size)#, random_init=True)  #  random_init=True gives me better Val scores

        # define some more size variables for shrinking & growing sizes in between encoder & decoder
        full_dim  = int(ft_size/2+1)   # this is the size of the output from the FNNAnalysis routine
        med_dim   = nearest_even(full_dim / ratio)
        small_dim = nearest_even(med_dim / ratio)

        #print("chunk_size, mid_size, ft_size = ",chunk_size, mid_size, ft_size)
        #print("full_dim, med_dim, small_dim = ",full_dim, med_dim, small_dim)

        self.shrink_r  = nn.Linear(full_dim,  med_dim, bias=False)
        self.shrink2_r = nn.Linear(med_dim,   small_dim, bias=False)
        self.shrink_i  = nn.Linear(full_dim,  med_dim, bias=False)   # separate weights for imag
        self.shrink2_i = nn.Linear(med_dim,   small_dim, bias=False)

        #self.dense = nn.Linear(self.small_dim, self.small_dim, bias=False)  # not needed
        self.grow2_r   = nn.Linear(small_dim, med_dim, bias=False)
        self.grow_r    = nn.Linear(med_dim,   full_dim, bias=False)

        self.grow2_i   = nn.Linear(small_dim, med_dim, bias=False)
        self.grow_i    = nn.Linear(med_dim,   full_dim, bias=False)

        self.mapskip_r  = nn.Linear(2*full_dim,  full_dim, bias=False)  # maps concatenated skip connection to 'regular' size
        self.mapskip2_r = nn.Linear(2*med_dim,   med_dim, bias=False)

        self.mapskip_i  = nn.Linear(2*full_dim,  full_dim, bias=False)  # maps concatenated skip connection to 'regular' size
        self.mapskip2_i = nn.Linear(2*med_dim,   med_dim, bias=False)

        self.bigskip = nn.Linear(2*mid_size,   mid_size, bias=False)

        #self.finalskip = nn.Linear(2*chunk_size,   chunk_size, bias=False)  # does not fit in CUDA memory

        self.act     = nn.LeakyReLU()   # Tried ReLU, ELU, SELU; Leaky seems to work best.  SELU yields instability

    def forward(self, input_var, skips=(2,3,4)):    # my trials indicate skips=(2,3) works best. (1,2,3) has more noise, (3) converges slower, no skips converges slowest
        y_orig = input_var
        y_s = self.act( self.front_shrink(y_orig) )     # _s to prepare for skip connection "skips=3"
        y = self.act( self.front_shrink2(y_s) )

        y = y.unsqueeze(0)   # hack to deal with batch size of 1 right now; TODO: remove this
        real_s, imag_s = self.encoder(y)                                                        # _s to prep for skip connection skips=2
        real_s2, imag_s2 = self.act( self.shrink_r(real_s) ),  self.act( self.shrink_i(imag_s) )    # _s2 for other skip connection  skips=1

        real, imag = self.act( self.shrink2_r(real_s2) ), self.act( self.shrink2_i(imag_s2) )
        real, imag = self.act( self.grow2_r(real) ),   self.act( self.grow2_i(imag) )

        if (1 in skips):  # this one has no discernable effect; but does yield more noise
            real, imag = self.mapskip2_r(torch.cat((real, real_s2), 2)), self.mapskip2_i(torch.cat((imag, imag_s2), 2))   # make skip connection

        real, imag = self.act( self.grow_r(real) ),    self.act( self.grow_i(imag) )

        if (2 in skips):
            real, imag = self.mapskip_r(torch.cat((real, real_s), 2)), self.mapskip_i(torch.cat((imag, imag_s), 2))   # make skip connection

        y = self.decoder(real, imag)
        y = y.squeeze(0)
        y = self.act(self.back_grow2(y))
        if (3 in skips):
            y = self.bigskip( torch.cat((y, y_s), 1) )
        y = self.back_grow(y)
        #if (4 in skips):
        #    y = self.finalskip( torch.cat((y, y_orig), 1) )  # too memory-intensive for large audio clips
        return y

# reuses same weights for real & imag
class SpecShrinkGrow_reuse(nn.Module):  # spectral bottleneck encoder
    # This model first projects the input audio to smaller sizes, using some 'dense' layers,
    #  Then calls the FNN analysis routine, does show shinking & growing, calls FNNSynthesis,
    #  and then expands back out to the original audio size.
    #  Two skip connections are present; more or fewer produced worse results that this.
    #  For a diagram of thi smodel, see ../docs/model_diagram.png
    #
    def __init__(self, chunk_size=1024):
        super(SpecShrinkGrow, self).__init__()


        ratio = 3   # Reduction ratio. 3 works about as well as 2.  4 seems to be too much
                    # one problem with ratio of 3 is it can produce odd sizes that don't split the data evenly between two GPUs
        mid_size = nearest_even(chunk_size/ratio)
        ft_size = nearest_even(mid_size /ratio)  # previously used ft_size = chunk_size, but that was limiting GPU memory usage

        # these shrink & grow routines are just to try to decrease encoder-decoder GPU memory usage
        self.front_shrink = nn.Linear(chunk_size, mid_size, bias=False)
        self.front_shrink2 = nn.Linear(mid_size, ft_size, bias=False)
        self.back_grow2    = nn.Linear(ft_size, mid_size, bias=False)
        self.back_grow    = nn.Linear(mid_size, chunk_size, bias=False)

        # the "FFT" routines
        self.encoder = front_end.FNNAnalysis(ft_size=ft_size)   # gives matrix mult. size mismatches
        self.decoder = front_end.FNNSynthesis(ft_size=ft_size)#, random_init=True)  #  random_init=True gives me better Val scores

        # define some more size variables for shrinking & growing sizes in between encoder & decoder
        full_dim  = int(ft_size/2+1)   # this is the size of the output from the FNNAnalysis routine
        med_dim   = nearest_even(full_dim / ratio)
        small_dim = nearest_even(med_dim / ratio)

        #print("chunk_size, mid_size, ft_size = ",chunk_size, mid_size, ft_size)
        #print("full_dim, med_dim, small_dim = ",full_dim, med_dim, small_dim)

        self.shrink  = nn.Linear(full_dim,  med_dim, bias=False)
        self.shrink2 = nn.Linear(med_dim,   small_dim, bias=False)
        #self.dense = nn.Linear(self.small_dim, self.small_dim, bias=False)  # not needed
        self.grow2   = nn.Linear(small_dim, med_dim, bias=False)
        self.grow    = nn.Linear(med_dim,   full_dim, bias=False)

        self.mapskip  = nn.Linear(2*full_dim,  full_dim, bias=False)  # maps concatenated skip connection to 'regular' size
        self.mapskip2 = nn.Linear(2*med_dim,   med_dim, bias=False)
        self.bigskip = nn.Linear(2*mid_size,   mid_size, bias=False)
        #self.finalskip = nn.Linear(2*chunk_size,   chunk_size, bias=False)  # does not fit in CUDA memory

        self.act     = nn.LeakyReLU()   # Tried ReLU, ELU, SELU; Leaky seems to work best.  SELU yields instability

    def forward(self, input_var, skips=(2,3,4)):    # my trials indicate skips=(2,3) works best. (1,2,3) has more noise, (3) converges slower, no skips converges slowest
        y_orig = input_var
        y_s = self.act( self.front_shrink(y_orig) )     # _s to prepare for skip connection "skips=3"
        y = self.act( self.front_shrink2(y_s) )

        y = y.unsqueeze(0)   # hack to deal with batch size of 1 right now; TODO: remove this
        real_s, imag_s = self.encoder(y)                                                        # _s to prep for skip connection skips=2
        real_s2, imag_s2 = self.act( self.shrink(real_s) ),  self.act( self.shrink(imag_s) )    # _s2 for other skip connection  skips=1

        real, imag = self.act( self.shrink2(real_s2) ), self.act( self.shrink2(imag_s2) )
        real, imag = self.act( self.grow2(real) ),   self.act( self.grow2(imag) )

        if (1 in skips):  # this one has no discernable effect; but does yield more noise
            real, imag = self.mapskip2(torch.cat((real, real_s2), 2)), self.mapskip2(torch.cat((imag, imag_s2), 2))   # make skip connection

        real, imag = self.act( self.grow(real) ),    self.act( self.grow(imag) )

        if (2 in skips):
            real, imag = self.mapskip(torch.cat((real, real_s), 2)), self.mapskip(torch.cat((imag, imag_s), 2))   # make skip connection

        y = self.decoder(real, imag)
        y = y.squeeze(0)
        y = self.act(self.back_grow2(y))
        if (3 in skips):
            y = self.bigskip( torch.cat((y, y_s), 1) )
        y = self.back_grow(y)
        #if (4 in skips):
        #    y = self.finalskip( torch.cat((y, y_orig), 1) )  # too memory-intensive for large audio clips
        return y



class SpecEncDec_old(nn.Module):  # old version; unused
    def __init__(self, ft_size=1024):          # ft_size must = 'chunk_size' elsewhere in code
        super(SpecEncDec_old, self).__init__()
        self.ft_size = ft_size
        self.w_size = self.ft_size * 2
        self.hop_size = self.ft_size
        #self.encoder = front_end.Analysis(ft_size=self.ft_size, w_size=self.w_size, hop_size=self.hop_size)
        #self.decoder = front_end.Synthesis(ft_size=self.ft_size, w_size=self.w_size, hop_size=self.hop_size)


        self.encoder = front_end.FNNAnalysis(ft_size=ft_size)   # gives matrix mult. size mismatches
        self.decoder = front_end.FNNSynthesis(ft_size=ft_size)

        #define some other other layers which we may or may not use (just messing around)
        self.cnn = CNNAuto(self.ft_size)
        self.full_dim = int(ft_size/2 +1)
        self.dense = nn.Linear(self.full_dim, self.full_dim)

        #self.batch = nn.BatchNorm2d(500)
        #proj_dim = int(self.full_dim*1.5)
        #self.proj = nn.Linear(self.full_dim, proj_dim )
        #self.deproj = nn.Linear( proj_dim, self.full_dim)

        self.act = nn.LeakyReLU()   # ReLU, LeakyReLU, ELU & SELU all yield similar performance in my tets so far
        #self.filter = LowPassLayer(self.ft_size)
        #self.dropout = torch.nn.Dropout2d(p=0.1)

    def forward(self, input_var):
        y = self.encoder(input_var)

        #y = self.proj(y)
        #y = self.act(y)
        #y = self.dropout(y)
        #middle = self.dense(x_ft)
        #middle = self.act(middle)  # nonlinear activation function

        # CNN version  works much more efficiently than fully-connected layers
        y = y.unsqueeze(1)             # add one array dimension at index 1
        y = self.cnn(y)                # added a new set of layer(s)
        y = y.squeeze(1)       # remove extra array dimension at index 1
        #y = self.dropout(y)

        #y = self.deproj(y)
        y = self.act(y)

        #middle = self.filter(middle)
        wave_form = self.decoder(y)
        return wave_form


# my attempt at a Seq2Seq model.  TODO: Fix problem: Runs of memory.
class Seq2Seq(nn.Module):
    # TODO: Try adding Attention
    def __init__(self, input_size=8192, hidden_size=2048):
        super(Seq2Seq, self).__init__()
        output_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.act = nn.LeakyReLU()          # results not strongly dependent on act. function
        self.embed = nn.Linear(input_size, hidden_size)
        self.de_embed = nn.Linear(hidden_size, output_size)
        self.hidden = self.initHidden()

    def forward(self, input_var):
        batch_size = input_var.size(0)
        time_domain_samples = input_var.size(1)
        input = input_var.view(batch_size, 1, time_domain_samples)

        projection = self.embed(input)
        projection = self.act(projection)
        representation, self.hidden = self.gru(projection, self.hidden)
        derep, self.hidden = self.gru(representation, self.hidden)
        output = self.de_embed(derep)
        output = output.view(batch_size, time_domain_samples)
        return output

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if torch.has_cudnn:
            return result.cuda()
        else:
            return result


# From  https://gist.github.com/lirnli/4282fcdfb383bb160cacf41d8c783c70
#  One hot encoding for WaveNet (below)
class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot,self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth).cuda()

    def forward(self, X_in):
        #print("X_in.squeeze(0).data = ",X_in.squeeze(0).data)
        #print("OneHot forward: X_in = ",X_in)
        return Variable(self.ones.index_select(0,X_in.squeeze(0).data))
    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

# From  https://gist.github.com/lirnli/4282fcdfb383bb160cacf41d8c783c70
# This assumes that the input and target signals have been mu-law companded
class WaveNet(nn.Module):
    def __init__(self, mu=256,n_residue=32, n_skip= 512, dilation_depth=10, n_repeat=5):
        # mu: audio quantization size
        # n_residue: residue channels
        # n_skip: skip channels
        # dilation_depth & n_repeat: dilation layer setup
        super(WaveNet, self).__init__()
        print("Initializing WaveNet")
        self.dilation_depth = dilation_depth
        dilations = self.dilations = [2**i for i in range(dilation_depth)] * n_repeat
        self.one_hot = One_Hot(mu).cuda()
        self.from_input = nn.Conv1d(in_channels=mu, out_channels=n_residue, kernel_size=1)
        self.conv_sigmoid = nn.ModuleList([nn.Conv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=2, dilation=d)
                         for d in dilations])
        self.conv_tanh = nn.ModuleList([nn.Conv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=2, dilation=d)
                         for d in dilations])
        self.skip_scale = nn.ModuleList([nn.Conv1d(in_channels=n_residue, out_channels=n_skip, kernel_size=1)
                         for d in dilations])
        self.residue_scale = nn.ModuleList([nn.Conv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=1)
                         for d in dilations])
        self.conv_post_1 = nn.Conv1d(in_channels=n_skip, out_channels=n_skip, kernel_size=1)
        self.conv_post_2 = nn.Conv1d(in_channels=n_skip, out_channels=mu, kernel_size=1)

    def forward(self, input):
        print("  wavenet forward: input = ",input)
        output = self.preprocess(input)
        skip_connections = [] # save for generation purposes
        for s, t, skip_scale, residue_scale in zip(self.conv_sigmoid, self.conv_tanh, self.skip_scale, self.residue_scale):
            output, skip = self.residue_forward(output, s, t, skip_scale, residue_scale)
            skip_connections.append(skip)
        # sum up skip connections
        output = sum([s[:,:,-output.size(2):] for s in skip_connections])
        output = self.postprocess(output)
        return output

    def preprocess(self, input):
        output = self.one_hot(input).unsqueeze(0).transpose(1,2)
        output = self.from_input(output)
        return output

    def postprocess(self, input):
        output = nn.functional.elu(input)
        output = self.conv_post_1(output)
        output = nn.functional.elu(output)
        output = self.conv_post_2(output).squeeze(0).transpose(0,1)
        return output

    def residue_forward(self, input, conv_sigmoid, conv_tanh, skip_scale, residue_scale):
        output = input
        output_sigmoid, output_tanh = conv_sigmoid(output), conv_tanh(output)
        output = nn.functional.sigmoid(output_sigmoid) * nn.functional.tanh(output_tanh)
        skip = skip_scale(output)
        output = residue_scale(output)
        output = output + input[:,:,-output.size(2):]
        return output, skip

    def generate_slow(self, input, n=100):
        res = input.data.tolist()
        for _ in range(n):
            x = Variable(torch.LongTensor(res[-sum(self.dilations)-1:]))
            y = self.forward(x)
            _, i = y.max(dim=1)
            res.append(i.data.tolist()[-1])
        return res

    def generate(self, input=None, n=100, temperature=None, estimate_time=False):
        ## prepare output_buffer
        output = self.preprocess(input)
        output_buffer = []
        for s, t, skip_scale, residue_scale, d in zip(self.conv_sigmoid, self.conv_tanh, self.skip_scale, self.residue_scale, self.dilations):
            output, _ = self.residue_forward(output, s, t, skip_scale, residue_scale)
            sz = 1 if d==2**(self.dilation_depth-1) else d*2
            output_buffer.append(output[:,:,-sz-1:-1])
        ## generate new
        res = input.data.tolist()
        for i in range(n):
            output = Variable(torch.LongTensor(res[-2:])).cuda()
            output = self.preprocess(output)
            output_buffer_next = []
            skip_connections = [] # save for generation purposes
            for s, t, skip_scale, residue_scale, b in zip(self.conv_sigmoid, self.conv_tanh, self.skip_scale, self.residue_scale, output_buffer):
                output, residue = self.residue_forward(output, s, t, skip_scale, residue_scale)
                output = torch.cat([b, output], dim=2)
                skip_connections.append(residue)
                if i%100==0:
                    output = output.clone()
                output_buffer_next.append(output[:,:,-b.size(2):])
            output_buffer = output_buffer_next
            output = output[:,:,-1:]
            # sum up skip connections
            output = sum(skip_connections)
            output = self.postprocess(output)
            if temperature is None:
                _, output = output.max(dim=1)
            else:
                output = output.div(temperature).exp().multinomial(1).squeeze()
            res.append(output.data[-1])
        return res




# visualize aspects of the model's internal state
def model_viz(model, outfileprefix):
    #print("model_viz:  ")

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

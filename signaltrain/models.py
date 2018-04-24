

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


class SpecShrinkGrow(nn.Module):  # spectral bottleneck encoder
    def __init__(self, chunk_size=1024):
        super(SpecShrinkGrow, self).__init__()

        ft_size = int(chunk_size /3)  # previously used ft_size = chunk_size, but that was limiting GPU memory usage

        # these shrink & grow routines are just to try to decrease encoder-decoder GPU memory usage
        self.front_shrink = nn.Linear(chunk_size, ft_size, bias=False)
        self.back_grow    = nn.Linear(ft_size, chunk_size, bias=False)

        # the "FFT" routines
        self.encoder = front_end.FNNAnalysis(ft_size=ft_size)   # gives matrix mult. size mismatches
        self.decoder = front_end.FNNSynthesis(ft_size=ft_size, random_init=True)  #  random_init=True gives me better Val scores

        # define some more size variables for shrinking & growing sizes in between encoder & decoder
        full_dim  = int(ft_size/2 +1)   # this is the size of the output from the FNNAnalysis routine
        ratio = 3   # Reduction ratio. Also tried 2; 3 works about as well as 2.  4 seems to be too much
        med_dim   = int(full_dim / ratio)
        small_dim = int(med_dim / ratio)

        self.shrink  = nn.Linear(full_dim,  med_dim, bias=False)
        self.shrink2 = nn.Linear(med_dim,   small_dim, bias=False)
        #self.dense = nn.Linear(self.small_dim, self.small_dim, bias=False)  # not needed
        self.grow2   = nn.Linear(small_dim, med_dim, bias=False)
        self.grow    = nn.Linear(med_dim,   full_dim, bias=False)
        self.act     = nn.LeakyReLU()   # Tried ReLU, ELU, SELU; Leaky seems to work best.  SELU yields instability

    def forward(self, input_var):
        y = input_var
        y = self.front_shrink(y)
        y = self.act(y)
        y = y.unsqueeze(0)   # hack to deal with batch size of 1 right now; TODO: remove this
        real, imag = self.encoder(y)
        real, imag = self.act( self.shrink(real) ),  self.act( self.shrink(imag) )
        real, imag = self.act( self.shrink2(real) ), self.act( self.shrink2(imag) )
        real, imag = self.act( self.grow2(real) ),   self.act( self.grow2(imag) )
        real, imag = self.act( self.grow(real) ),    self.act( self.grow(imag) )
        y = self.decoder(real, imag)
        y = y.squeeze(0)
        y = self.back_grow(y)
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

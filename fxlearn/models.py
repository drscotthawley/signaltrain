__author__ = 'S.H. Hawley'

# imports
import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import sys
from . import transforms as front_end


class CNNAuto(nn.Module):   # 'convolutional autoencoder'
    def __init__(self):
        super(CNNAuto, self).__init__()
        self.io_channels = 1   # mono for now
        self.feature_channels = 4  # chose this arbitrarily. nothing sacred about this number!
        self.kernel_size = 7
        self.pad = int(self.kernel_size/2)
        self.scale_factor = 2
        self.layer_in = nn.Sequential(     # generates feature_channels times more data
            nn.Conv2d(self.io_channels, self.feature_channels, kernel_size=self.kernel_size, padding=self.pad),
            #nn.BatchNorm2d(self.feature_channels),
            nn.SELU()
            )

        self.layer_mid = nn.Sequential(    # keeps number of feature channels the same
            nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=self.kernel_size, padding=self.pad),
            #nn.BatchNorm2d(self.feature_channels),
            nn.SELU()
            )

        # Pooling and Un-Pooling
        self.layer_shrink =  nn.Sequential( nn.MaxPool2d(self.scale_factor) )
        self.layer_grow =  nn.Sequential( nn.Upsample(scale_factor=self.scale_factor, mode='bilinear') )

        self.layer_out = nn.Sequential(     # combines feature channels back to original # of channels
            nn.Conv2d(self.feature_channels, self.io_channels, kernel_size=self.kernel_size, padding=self.pad),
            #nn.BatchNorm2d(self.io_channels),
            nn.SELU())

    def forward(self, x):
        y = self.layer_in(x)
        '''  shrinking & growing is causing problems; disabled for now
        y = self.layer_shrink(y)        #   \      /
        y = self.layer_mid(y)
        y = self.layer_shrink(y)        #    \   /
        '''
        y = self.layer_mid(y)
        y = self.layer_mid(y)

        '''
        y = self.layer_grow(y)          #    /   \
        y = self.layer_mid(y)
        y = self.layer_grow(y)          #  /      \
        '''
        y = self.layer_out(y)
        return y

# TODO: doesn't work yet. Unused.
class LowPassLayer(nn.Module):
    ''' Description:  This is a tunable 'reverse sigmoid' soft mask, which will get multiplied (element-wise)
                      by the (vertical columns of) frequency spectrum, to serve as a low-pass filter.
                      It has two parameters: the central frequency (from 0...1, scales with size of input)
                      and a steepness parameter (also between -1.0=flat and 1.0=vertical)
                      TODO: how do we best bound these to positive values?

        Purpose:      Much of the loss-reduction in the final network output gets bound up in damping high-frequency noise.
                      In theory we could let the network do this eventually, but this layer is to help converge faster.

        Note:         Demo & discussion, see ../docs/LowPassLayer.ipynb
                      Layer-writing guide at https://discuss.pytorch.org/t/how-to-define-a-new-layer-with-autograd/351

        TODO:          Make indices work with batch_size
    '''
    def __init__(self, bins):
        super(LowPassLayer, self).__init__()
        self.m = nn.Parameter(torch.zeros(1))      # steepness, tunable parameter, default is 0
        self.b = nn.Parameter(torch.zeros(1))      # center, tunable parameter, default is 0
        self.eps = 0.1
        self.bins = bins
        self.x = Variable(torch.linspace(0, bins-1, num=bins))

    def forward(self, in_stft):
        # given matrix A (n x m) and vector v (n elements), the operation we want is (A.T * v).T
        #   This method is also the fastest: https://stackoverflow.com/questions/18522216/multiplying-across-in-a-numpy-array

        m = self.m.expand_as(self.x)     # make it a 1-D vector of repeated numbers
        b = self.b.expand_as(self.x)
        mask = 1/(1+torch.exp( (1+m)/(1-m+self.eps)*10*(self.x/self.bins-(b+0.5)) )  )
        out_stft = torch.transpose( torch.mm( torch.transpose(in_stft), mask ) )
        return out_stft


class SpecEncDec(nn.Module):  # 'spectral encoder-decoder'
    def __init__(self):
        super(SpecEncDec, self).__init__()
        self.ft_size = 4096
        self.w_size = self.ft_size * 2
        self.hop_size = self.ft_size
        self.encoder = front_end.Analysis(ft_size=self.ft_size, w_size=self.w_size, hop_size=self.hop_size)
        self.decoder = front_end.Synthesis(ft_size=self.ft_size, w_size=self.w_size, hop_size=self.hop_size)
        #self.encoder = front_end.FNNAnalysis()   # gives matrix mult. size mismatches
        #self.decoder = front_end.FNNSynthesis()

        #define some other other layers which we may or may not use (just messing around)
        self.cnn = CNNAuto()
        self.full_dim = self.ft_size
        self.dense = nn.Linear(self.full_dim, self.full_dim)
        self.act = nn.SELU()   # ReLU, LeakyReLU, ELU & SELU all yield similar performance in my tets so far

        '''

        self.small_dim = self.ft_size

        self.shrink = nn.Linear(self.full_dim, self.small_dim)  # fully connected layer

        self.small = nn.Linear(self.small_dim, self.small_dim)
        self.grow = nn.Linear(self.small_dim, self.full_dim)  # fully connected layer
        '''


    # Here's where we run the network forward (PyTorch takes care of backward)
    def forward(self, input_var):
        x_ft = self.encoder(input_var)

        #middle = self.dense(x_ft)
        #middle = self.act(middle)  # nonlinear activation function

        # CNN version  works much more efficiently than fully-connected layers
        x_ft = x_ft.unsqueeze(1)             # add one array dimension at index 1
        middle = self.cnn(x_ft)                # added a new set of layer(s)
        middle = middle.squeeze(1)       # remove extra array dimension at index 1

        wave_form = self.decoder(middle)
        return wave_form



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

#  EOF

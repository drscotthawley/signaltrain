
# -*- coding: utf-8 -*-
__author__ = 'S. Venkataramani, S.I. Mimilakis'
__copyright__ = 'ECE Illinois, MacSeNet'

# imports
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import cosine

class FNNAnalysis(nn.Module):
    """
        Class for building the analysis part
        of the Front-End ('Fe').
    """

    def __init__(self, ft_size=1024):
        super(FNNAnalysis, self).__init__()

        # Parameters
        self.batch_size = None
        self.time_domain_samples = None
        self.sz = ft_size
        self.half_N = int(self.sz / 2 + 1)

        # Analysis
        self.fnn_analysis_real = nn.Linear(self.sz, self.sz, bias=False)
        self.fnn_analysis_imag = nn.Linear(self.sz, self.sz, bias=False)

        # Custom Initialization with Fourier matrix
        self.initialize()

    def initialize(self):
        f_matrix = np.fft.fft(np.eye(self.sz))

        f_matrix_real = (np.real(f_matrix)).astype(np.float32)
        f_matrix_imag = (np.imag(f_matrix)).astype(np.float32)

        # Note from Scott: can CUDA-ify later by calling, e.g., model.cuda()
        self.fnn_analysis_real.weight.data.copy_(torch.from_numpy(f_matrix_real))
        self.fnn_analysis_imag.weight.data.copy_(torch.from_numpy(f_matrix_imag))

    def forward(self, wave_form):
        an_real = self.fnn_analysis_real(wave_form)[:, :, :self.half_N]
        an_imag = self.fnn_analysis_imag(wave_form)[:, :, :self.half_N]

        return an_real, an_imag


class FNNSynthesis(nn.Module):
    """
        Class for building the synthesis part
        of the Front-End ('Fe').
    """    

    def __init__(self, ft_size=1024, random_init=False):
        super(FNNSynthesis, self).__init__()

        # Parameters
        self.batch_size = None
        self.time_domain_samples = None
        self.sz = ft_size
        self.half_N = int(self.sz / 2 + 1)

        # Synthesis
        self.fnn_synthesis_real = nn.Linear(self.sz, self.sz, bias=False)
        self.fnn_synthesis_imag = nn.Linear(self.sz, self.sz, bias=False)

        # Tanh
        self.tanh = torch.nn.Tanh()

        if random_init:
            # Random Initialization
            self.initialize_random()
        else:
            # Custom Initialization with Fourier matrix
            self.initialize()

    def initialize(self):
        print('Initializing with Fourier bases')
        f_matrix = np.fft.fft(np.eye(self.sz), norm='ortho')

        f_matrix_real = (np.real(f_matrix)).astype(np.float32)
        f_matrix_imag = (np.imag(f_matrix)).astype(np.float32)

        self.fnn_synthesis_real.weight.data.copy_(torch.from_numpy(f_matrix_real.T))
        self.fnn_synthesis_imag.weight.data.copy_(torch.from_numpy(f_matrix_imag.T))

    def initialize_random(self):
        print('Initializing randomly')
        nn.init.xavier_uniform(self.fnn_synthesis_real.weight)
        nn.init.xavier_uniform(self.fnn_synthesis_imag.weight)

    def forward(self, real, imag):
        real = torch.cat((real, FNNSynthesis.flip(real[:, :, 1:-1].contiguous(), 2)), 2)
        imag = torch.cat((imag, FNNSynthesis.flip(-imag[:, :, 1:-1].contiguous(), 2)), 2)

        wave_form = self.tanh(self.fnn_synthesis_real(real) + self.fnn_synthesis_imag(imag))
        return wave_form

    @staticmethod
    def flip(x, dim):
        # https://github.com/pytorch/pytorch/issues/229
        xsize = x.size()
        dim = x.dim() + dim if dim < 0 else dim
        x = x.view(-1, *xsize[dim:])
        x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                         -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
        return x.view(xsize)

# EOF

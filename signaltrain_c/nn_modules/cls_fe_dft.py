# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import numpy as np
import torch
import torch.nn as nn
from scipy import signal as sig


class Analysis(nn.Module):
    """
        Class for building the analysis part
        of the Front-End ('Fe').
    """
    def __init__(self, ft_size=1024, hop_size=384):
        super(Analysis, self).__init__()

        # Parameters
        self.batch_size = None
        self.time_domain_samples = None
        self.sz = ft_size
        self.hop = hop_size
        self.half_N = int(self.sz/2. + 1)

        # Analysis 1D CNN
        self.conv_analysis_real = nn.Conv1d(1, self.sz, self.sz,
                                            padding=self.sz, stride=self.hop, bias=False)
        self.conv_analysis_imag = nn.Conv1d(1, self.sz, self.sz,
                                            padding=self.sz, stride=self.hop, bias=False)

        # Custom Initialization with Fourier matrix
        self.initialize()

    def initialize(self):
        f_matrix = np.fft.fft(np.eye(self.sz), norm='ortho')
        w = sig.hamming(self.sz)

        f_matrix_real = (np.real(f_matrix) * w).astype(np.float32)
        f_matrix_imag = (np.imag(f_matrix) * w).astype(np.float32)

        if torch.has_cudnn:
            self.conv_analysis_real.weight.data.copy_(torch.from_numpy(f_matrix_real[:, None, :]).cuda())
            self.conv_analysis_imag.weight.data.copy_(torch.from_numpy(f_matrix_imag[:, None, :]).cuda())
        else:
            self.conv_analysis_real.weight.data.copy_(torch.from_numpy(f_matrix_real[:, None, :]))
            self.conv_analysis_imag.weight.data.copy_(torch.from_numpy(f_matrix_imag[:, None, :]))

    def forward(self, wave_form):
        batch_size = wave_form.size(0)
        time_domain_samples = wave_form.size(1)

        wave_form = wave_form.view(batch_size, 1, time_domain_samples)
        an_real = self.conv_analysis_real(wave_form).transpose(1, 2)[:, :, :self.half_N]
        an_imag = self.conv_analysis_imag(wave_form).transpose(1, 2)[:, :, :self.half_N]

        return an_real, an_imag


class Synthesis(nn.Module):
    """
        Class for building the synthesis part
        of the Front-End ('Fe').
    """

    def __init__(self, ft_size=1024, hop_size=384):
        super(Synthesis, self).__init__()

        # Parameters
        self.batch_size = None
        self.time_domain_samples = None
        self.sz = ft_size
        self.hop = hop_size
        self.half_N = int(self.sz / 2 + 1)

        # Synthesis 1D CNN
        self.conv_synthesis_real = nn.ConvTranspose1d(self.sz, 1, self.sz,
                                                      padding=0, stride=self.hop, bias=False)

        self.conv_synthesis_imag = nn.ConvTranspose1d(self.sz, 1, self.sz,
                                                      padding=0, stride=self.hop, bias=False)

        # Custom Initialization with Fourier matrix
        self.initialize()

    def initialize(self):
        f_matrix = np.fft.fft(np.eye(self.sz), norm='ortho')
        w = Synthesis.GLA(self.sz, self.hop, self.sz)

        f_matrix_real = (np.real(f_matrix) * w).astype(np.float32)
        f_matrix_imag = (np.imag(f_matrix) * w).astype(np.float32)

        if torch.has_cudnn:
            self.conv_synthesis_real.weight.data.copy_(torch.from_numpy(f_matrix_real[:, None, :]).cuda())
            self.conv_synthesis_imag.weight.data.copy_(torch.from_numpy(f_matrix_imag[:, None, :]).cuda())

        else:
            self.conv_synthesis_real.weight.data.copy_(torch.from_numpy(f_matrix_real[:, None, :]))
            self.conv_synthesis_imag.weight.data.copy_(torch.from_numpy(f_matrix_imag[:, None, :]))

    def forward(self, real, imag):
        real = torch.transpose(real, 1, 2)
        imag = torch.transpose(imag, 1, 2)

        real = torch.cat((real, Synthesis.flip(real[:, 1:-1, :].contiguous(), 1)), 1)
        imag = torch.cat((imag, Synthesis.flip(-imag[:, 1:-1, :].contiguous(), 1)), 1)

        wave_form = self.conv_synthesis_real(real) + self.conv_synthesis_imag(imag)
        wave_form = wave_form[:, :, self.sz:-self.sz]

        return wave_form[:, 0, :]

    @staticmethod
    def flip(x, dim):
        # https://github.com/pytorch/pytorch/issues/229
        xsize = x.size()
        dim = x.dim() + dim if dim < 0 else dim
        x = x.view(-1, *xsize[dim:])
        x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                     -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(),
                                             :]
        return x.view(xsize)

    @staticmethod
    def GLA(wsz, hop, N=4096):
        """ LSEE-MSTFT algorithm for computing the synthesis window used in
        inverse STFT method.
        Args:
            wsz :   (int)    Synthesis window size
            hop :   (int)    Hop size
            N   :   (int)    DFT Size
        Returns :
            symw:   (array)  Synthesised windowing function
        References :
            [1] Daniel W. Griffin and Jae S. Lim, ``Signal estimation from modified short-time
            Fourier transform,'' IEEE Transactions on Acoustics, Speech and Signal Processing,
            vol. 32, no. 2, pp. 236-243, Apr 1984.
        """
        synw = sig.hamming(wsz)
        synwProd = synw ** 2.
        synwProd.shape = (wsz, 1)
        redundancy = wsz // hop
        env = np.zeros((wsz, 1))
        for k in range(-redundancy, redundancy + 1):
            envInd = (hop * k)
            winInd = np.arange(1, wsz + 1)
            envInd += winInd
            valid = np.where((envInd > 0) & (envInd <= wsz))
            envInd = envInd[valid] - 1
            winInd = winInd[valid] - 1
            env[envInd] += synwProd[winInd]

        synw = synw / env[:, 0]
        return synw


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
        f_matrix = np.fft.fft(np.eye(self.sz), norm='ortho')

        f_matrix_real = (np.real(f_matrix)).astype(np.float32)
        f_matrix_imag = (np.imag(f_matrix)).astype(np.float32)

        if torch.has_cudnn:
            self.fnn_analysis_real.weight.data.copy_(torch.from_numpy(f_matrix_real).cuda())
            self.fnn_analysis_imag.weight.data.copy_(torch.from_numpy(f_matrix_imag).cuda())
        else:
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

        if torch.has_cudnn:
            self.fnn_synthesis_real.weight.data.copy_(torch.from_numpy(f_matrix_real.T).cuda())
            self.fnn_synthesis_imag.weight.data.copy_(torch.from_numpy(f_matrix_imag.T).cuda())

        else:
            self.fnn_synthesis_real.weight.data.copy_(torch.from_numpy(f_matrix_real.T))
            self.fnn_synthesis_imag.weight.data.copy_(torch.from_numpy(f_matrix_imag.T))

    def initialize_random(self):
        print('Initializing randomly')
        nn.init.xavier_uniform(self.fnn_synthesis_real.weight)
        nn.init.xavier_uniform(self.fnn_synthesis_imag.weight)

    def forward(self, real, imag):
        real = torch.cat((real, Synthesis.flip(real[:, :, 1:-1].contiguous(), 2)), 2)
        imag = torch.cat((imag, Synthesis.flip(-imag[:, :, 1:-1].contiguous(), 2)), 2)

        wave_form = self.fnn_synthesis_real(real) + self.fnn_synthesis_imag(imag)
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

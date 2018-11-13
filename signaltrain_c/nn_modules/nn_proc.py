
# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import torch
import torch.nn as nn
from .cls_fe_dft import Analysis, Synthesis

class AutoEncoder(nn.Module):
    def __init__(self, T=25, R=2, K=3):
        super(AutoEncoder, self).__init__()

        # Parameters
        self._T = T
        self._R = R
        self._K = K  # K = number of knobs

        # Analysis
        self.fnn_enc = nn.Linear(self._T, self._R, bias=True)
        self.fnn_enc2 = nn.Linear(self._R, self._R//2, bias=True)
        self.fnn_enc3 = nn.Linear(self._R//2, self._R//4, bias=True)
        self.fnn_addknobs = nn.Linear(self._R//4 + self._K, self._R//4, bias=True)
        self.fnn_dec3 = nn.Linear(self._R//4, self._R//2, bias=True)
        self.fnn_dec2 = nn.Linear(self._R//2, self._R, bias=True)
        self.fnn_dec = nn.Linear(self._R, self._T, bias=True)

        # Activation functions
        self.relu = nn.LeakyReLU()

        self.initialize()

    def initialize(self):
        torch.nn.init.xavier_normal_(self.fnn_enc.weight)
        torch.nn.init.xavier_normal_(self.fnn_enc2.weight)
        torch.nn.init.xavier_normal_(self.fnn_enc3.weight)
        torch.nn.init.xavier_normal_(self.fnn_addknobs.weight)
        torch.nn.init.xavier_normal_(self.fnn_dec2.weight)
        torch.nn.init.xavier_normal_(self.fnn_dec3.weight)
        torch.nn.init.xavier_normal_(self.fnn_dec.weight)
        self.fnn_enc.bias.data.zero_()
        self.fnn_enc2.bias.data.zero_()
        self.fnn_enc3.bias.data.zero_()
        self.fnn_addknobs.bias.data.zero_()
        self.fnn_dec3.bias.data.zero_()
        self.fnn_dec2.bias.data.zero_()
        self.fnn_dec.bias.data.zero_()

    def forward(self, x_input, knobs, skip_connections='res'):
        x_input = x_input.transpose(2, 1)
        z = self.relu(self.fnn_enc(x_input))
        z = self.relu(self.fnn_enc2(z))
        zenc3 = self.relu(self.fnn_enc3(z))
        z = zenc3
        knobs_r = knobs.unsqueeze(1).repeat(1, z.size()[1], 1)  # repeat the knobs to make dimensions match
        z = self.relu( self.fnn_addknobs( torch.cat((z,knobs_r),2) ) )
        z = self.relu( self.fnn_dec3(z + zenc3))
        z = self.relu( self.fnn_dec2(z))
        if skip_connections == 'exp':           # Refering to an old AES paper for exponentiation
            z_a = torch.log(self.relu(self.fnn_dec(z)) + 1e-6) * torch.log(x_input + 1e-6)
            out = torch.exp(z_a)
        elif skip_connections == 'res':         # Refering to residual connections
            out = self.relu(self.fnn_dec(z) + x_input)
        elif skip_connections == 'sf':          # Skip-filter
            out = self.relu(self.fnn_dec(z)) * x_input
        else:
            out = self.relu(self.fnn_dec(z))

        result = out.transpose(2, 1)
        return result


class MPAEC(nn.Module):  # mag-phase autoencoder
    """
        Class for building the analysis part
        of the Front-End ('Fe').
    """
    def __init__(self, expected_time_frames, ft_size=1024, hop_size=384, decomposition_rank=64):
        super(MPAEC, self).__init__()
        self.dft_analysis = Analysis(ft_size=ft_size, hop_size=hop_size)
        self.dft_synthesis = Synthesis(ft_size=ft_size, hop_size=hop_size)
        self.aenc = AutoEncoder(expected_time_frames, decomposition_rank)
        self.phs_aenc = AutoEncoder(expected_time_frames, 64)


    def clip_grad_norm_(self):
        torch.nn.utils.clip_grad_norm_(list(self.dft_analysis.parameters()) +
                                      list(self.dft_synthesis.parameters()),
                                      max_norm=1., norm_type=1)

    def forward(self, x_cuda, knobs_cuda):
        # trainable STFT, outputs spectrograms for real & imag parts
        x_real, x_imag = self.dft_analysis.forward(x_cuda/2)  # the /2 is to normalize
        # Magnitude-Phase computation
        mag = torch.norm(torch.cat((x_real.unsqueeze(0), x_imag.unsqueeze(0)), 0), 2, dim=0)
        phs = torch.atan2(x_imag, x_real+1e-6)

        # Processes Magnitude and phase individually
        mag_hat = self.aenc.forward(mag, knobs_cuda, skip_connections='sf')
        phs_hat = self.phs_aenc.forward(phs, knobs_cuda, skip_connections=False) + phs # <-- Slightly smoother convergence

        # Back to Real and Imaginary
        an_real = mag_hat * torch.cos(phs_hat)
        an_imag = mag_hat * torch.sin(phs_hat)

        # Forward synthesis pass
        x_hat = self.dft_synthesis.forward(an_real, an_imag)

        #x_hat += x_cuda   # No residual connection here -- makes it worse

        return 2*x_hat, 2*mag, 2*mag_hat

# EOF


# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis & S.H. Hawley'
__copyright__ = 'MacSeNet'

# imports
import torch
import numpy as np
import torch.nn as nn
try:   # because I can't manage to write relative imports that work both intra-package and inter-package
    from . cls_fe_dft import Analysis, Synthesis
except ImportError:
    from cls_fe_dft import Analysis, Synthesis
torch.backends.cudnn.benchmark = True   # makes Turing GPU ops faster! https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3


def freeze_layers(layers):
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = False

def unfreeze_layers(layers):
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = True


class AsymAutoEncoder(nn.Module):
    def __init__(self, T=25, R=64, K=3, OT=None, use_bias=True, use_dropout=False):
        super(AsymAutoEncoder, self).__init__()

        # Parameters
        self._T = T   # expected number of time frames
        self._R = R   # decomposition rank - (output) size of first encoded layer of activations
        self._K = K   # number of knobs
        if OT is None:  # expected number of output time frames
            self._OT = T
        else:
            self._OT = OT
        self.use_bias = use_bias
        self.use_dropout = use_dropout

        print("AsymAutoEncoder __init__: T, R, K, OT = ",T, R, K, OT)

        # Analysis
        rf = 2   # "reduction factor": size ratio between successive layers in autoencoder
        self.fnn_enc = nn.Linear(self._T, self._R, bias=self.use_bias)
        self.fnn_enc2 = nn.Linear(self._R, self._R//rf, bias=self.use_bias)
        self.fnn_enc3 = nn.Linear(self._R//rf, self._R//rf**2, bias=self.use_bias)
        self.fnn_enc4 = nn.Linear(self._R//rf**2, self._R//rf**2, bias=self.use_bias)

        self.fnn_addknobs = nn.Linear(self._R//rf**2 + self._K, self._R//rf**2, bias=self.use_bias)

        self.fnn_dec4 = nn.Linear(self._R//rf**2, self._R//rf**2, bias=self.use_bias) # repeat size, now with knobs catted
        self.fnn_dec3 = nn.Linear(self._R//rf**2, self._R//rf, bias=self.use_bias)
        self.fnn_dec2 = nn.Linear(self._R//rf, self._R, bias=self.use_bias)
        self.fnn_dec = nn.Linear(self._R, self._OT, bias=self.use_bias)

        self.layer_list = [self.fnn_enc, self.fnn_enc2, self.fnn_enc3, self.fnn_enc4,
            self.fnn_addknobs, self.fnn_dec4, self.fnn_dec3, self.fnn_dec2, self.fnn_dec]

        # Activation function(s)
        self.relu = nn.ELU()#  nn.LeakyReLU()
        #self.relu = nn.ReLU()

        # Dropout regularization
        if self.use_dropout: self.dropout = nn.Dropout2d(p=0.2)

        self.initialize()

    def initialize(self):
        for x in self.layer_list:
            torch.nn.init.xavier_normal_(x.weight)
            if self.use_bias:
                x.bias.data.zero_()

    def forward(self, x_input, knobs, skip_connections='res', return_acts=False):
        acts = []                                          # list of activations
        x_input = x_input.transpose(2, 1)
        z = self.relu(self.fnn_enc(x_input))
        if return_acts: acts.append(z)
        z = self.dropout(z) if self.use_dropout else z
        z = self.relu(self.fnn_enc2(z))
        if return_acts: acts.append(z)
        z = self.dropout(z) if self.use_dropout else z
        z = self.relu(self.fnn_enc3(z))
        if return_acts: acts.append(z)
        z = self.relu(self.fnn_enc4(z))
        if return_acts: acts.append(z)


        knobs_r = knobs.unsqueeze(1).repeat(1, z.size()[1], 1)  # repeat the knobs to make dimensions match
        catted = torch.cat((z,knobs_r),2)
        if return_acts: acts.append(catted)

        z = self.relu( self.fnn_addknobs( catted ) )
        if return_acts: acts.append(z)

        z = self.relu( self.fnn_dec4(z))
        if return_acts: acts.append(z)

        z = self.relu( self.fnn_dec3(z))
        if return_acts: acts.append(z)

        z = self.dropout(z) if self.use_dropout else z
        z = self.relu( self.fnn_dec2(z))
        if return_acts: acts.append(z)

        if skip_connections == 'exp':           # Refering to an old AES paper for exponentiation
            z_a = torch.log(self.relu(self.fnn_dec(z)) + 1e-6) * torch.log(x_input[:,-self._OT:,:] + 1e-6)
            out = torch.exp(z_a)
        elif skip_connections == 'res':         # Refering to residual connections
            out = self.relu(self.fnn_dec(z) + x_input[:,:,-self._OT:])
        elif skip_connections == 'sf':          # Skip-filter
            out = self.relu(self.fnn_dec(z)) * x_input[:,:,-self._OT:]
        else:
            out = self.relu(self.fnn_dec(z))
        out = self.dropout(out) if self.use_dropout else out
        if return_acts: acts.append(out)

        result = out.transpose(2, 1)

        if return_acts:
            return result, acts
        else:
            return result



'''
# currently CNN version is unused because it's slow

class cnnblock(nn.Module):
    def __init__(self, bn_size=1):
        super(cnnblock, self).__init__()
        self.myblock = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=1),
            nn.BatchNorm2d(bn_size)
        )
    def forward(self, input):
        return self.myblock(input)

class cnntransblock(nn.Module):
    def __init__(self, k=2, bn_size=1):
        super(cnntransblock, self).__init__()
        self.myblock = nn.Sequential(
            nn.ConvTranspose2d(1, 1, k, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(bn_size)
        )
    def forward(self, input):
        return self.myblock(input)



class AsymCNNAutoEncoder(nn.Module):
    def __init__(self, T=25, R=64, K=3, OT=None, use_bias=True, use_dropout=False):
        super(AsymCNNAutoEncoder, self).__init__()

        # Parameters
        self._T = T   # expected number of time frames
        self._R = R   # decomposition rank - (output) size of first encoded layer of activations
        self._K = K   # number of knobs
        if OT is None:  # expected number of output time frames
            self._OT = T
        else:
            self._OT = OT
        self.use_bias = use_bias

        print("AsymCNNAutoEncoder __init__: T, R, K, OT = ",T, R, K, OT)

        # Analysis
        rf = 2   # "reduction factor": size ratio between successive layers in autoencoder
        self.fnn_enc = nn.Linear(self._T, self._R, bias=self.use_bias)

        #self.fnn_enc2 = nn.Linear(self._R, self._R//rf, bias=self.use_bias)
        #self.fnn_enc3 = nn.Linear(self._R//rf, self._R//rf**2, bias=self.use_bias)
        #self.fnn_enc4 = nn.Linear(self._R//rf**2, self._R//rf**2, bias=self.use_bias)
        #self.fnn_enc = nn.Conv2d(1, 1, 3)
        self.fnn_enc2 = cnnblock()
        self.fnn_enc3 = cnnblock()
        self.fnn_enc4 = cnnblock()

        self.fnn_addknobs = nn.Linear(self._R//rf**3 + self._K, self._R//rf**2, bias=self.use_bias)
        self.fnn_backdown = nn.Linear(self._R//rf**2, self._R//rf**3, bias=self.use_bias)
        self.fnn_dec4 = cnntransblock()
        self.fnn_dec3 = cnntransblock()
        self.fnn_dec2 = cnntransblock(k=(3,2))
        self.outprep = nn.Linear(self._R*2+1, self._R, bias=self.use_bias)
        self.fnn_dec = nn.Linear(self._R, self._OT, bias=self.use_bias)

        self.layer_list = [self.fnn_enc, self.fnn_enc2, self.fnn_enc3, self.fnn_enc4,
            self.fnn_addknobs, self.fnn_dec4, self.fnn_dec3, self.fnn_dec2, self.fnn_dec]

        # Activation function(s)
        self.relu = nn.ELU()#  nn.LeakyReLU()
        #self.relu = nn.ReLU()

        self.initialize()

    def initialize(self):
        return # TODO: remove
        for x in self.layer_list:
            torch.nn.init.xavier_normal_(x.weight)
            if self.use_bias:
                x.bias.data.zero_()

    def forward(self, x_input, knobs, skip_connections='res', log=False, return_acts=False):
        x_input = x_input.transpose(2, 1)  # flip frequency & time (so we operate on time)

        # encoder
        if log: print("x_input.size = ",x_input.size())
        z = self.relu(self.fnn_enc(x_input))
        z = z.unsqueeze(1)  # add 1 "channel"
        if log: print("1 z.size = ",z.size())
        z = self.fnn_enc2(z)
        if log: print("2 z.size = ",z.size())
        z = self.fnn_enc3(z)
        if log: print("3 z.size = ",z.size())
        z = self.fnn_enc4(z)
        if log: print("4 z.size = ",z.size())

        z = z.squeeze(1)
        # middle of the autoencoder: merge the knobs
        knobs_r = knobs.unsqueeze(1).repeat(1, z.size()[1], 1)  # repeat the knobs to make dimensions match
        if log: print("knobs_r.size = ",knobs_r.size)
        catted = torch.cat((z,knobs_r),2)
        if log: print("catted.size = ",catted.size() )
        z = self.relu( self.fnn_addknobs( catted ) )
        z = z.unsqueeze(1)
        if log: print("5 knobs added: z.size() = ",z.size())
        z = self.fnn_backdown(z)
        if log: print("5.5 knobs added: z.size() = ",z.size())
        # decoder
        z = self.fnn_dec4(z)
        if log: print("6 z.size() = ",z.size())
        z = self.fnn_dec3(z)
        if log: print("7 z.size() = ",z.size())
        z = self.fnn_dec2(z)
        if log: print("8 z.size() = ",z.size())

        z = z.squeeze(1)
        if log: print("9 z.size() = ",z.size())

        # cnn's finished, ready for final Linear map

        if skip_connections == 'exp':           # Refering to an old AES paper for exponentiation
            z_a = torch.log(self.relu(self.fnn_dec(z)) + 1e-6) * torch.log(x_input[:,-self._OT:,:] + 1e-6)
            out = torch.exp(z_a)
        elif skip_connections == 'res':         # Refering to residual connections
            out = self.relu(self.fnn_dec(z) + x_input[:,:,-self._OT:])
        elif skip_connections == 'sf':          # Skip-filter
            out = self.relu(self.fnn_dec(z)) * x_input[:,:,-self._OT:]
        else:
            out = self.relu(self.fnn_dec(z))

        result = out.transpose(2, 1)
        return result
'''


class AsymMPAEC(nn.Module):
    """
        Asymmetric Magnitude-Phase AutoEncoder (with Knobs)
        'asymmetric' because output size != input size
        See st_model() below for generic calling

        Inputs:
           expected_time_frames:     Intended number of STFT time frames on input side
           output_tf:                Intended number of STFT time frames on output side
           n_knobs:                  Number of knobs to concatenate in the middle of the model
           ft_size:                  size of Fourier transform in STFT
           hop_size:                 hop size between STFT frames
           decomposition_rank:       size of first-smaller layer in autoencoder (we added more as we wrote this)
    """
    def __init__(self, expected_time_frames, ft_size=1024, hop_size=384,
        decomposition_rank=64, n_knobs=4, output_tf=None):
        super(AsymMPAEC, self).__init__()

        print("AsymMPAEC: expected_time_frames, ft_size, hop_size, decomposition_rank, n_knobs, output_tf = ", expected_time_frames, ft_size, hop_size, decomposition_rank, n_knobs, output_tf)
        if output_tf is None:
            self.output_tf = expected_time_frames
        else:
            self.output_tf = output_tf

        self.dft_analysis = Analysis(ft_size=ft_size, hop_size=hop_size)
        self.dft_synthesis = Synthesis(ft_size=ft_size, hop_size=hop_size)
        self.aenc = AsymAutoEncoder(T=expected_time_frames, R=decomposition_rank, K=n_knobs, OT=self.output_tf)
        self.phs_aenc = AsymAutoEncoder(T=expected_time_frames, R=decomposition_rank, K=n_knobs, OT=self.output_tf)
        #self.valve = nn.Parameter(torch.tensor([0.2,1.0]), requires_grad=True)  # "wet-dry mix"


    def reinitialize(self):  # randomly reassigns weights
        self.aenc.initialize()
        self.phs_aenc.initialize()

    def clip_grad_norm_(self):
        torch.nn.utils.clip_grad_norm_(list(self.dft_analysis.parameters()) +
                                      list(self.dft_synthesis.parameters()),
                                      max_norm=1., norm_type=1)


    def forward(self, x_cuda, knobs_cuda, return_acts=False):
        # trainable STFT, outputs spectrograms for real & imag parts
        x_real, x_imag = self.dft_analysis.forward(x_cuda/2)  # the /2 is cheap way to help us approach 'unit variaance' of -0.5 and .5
        # Magnitude-Phase computation
        mag = torch.norm(torch.cat((x_real.unsqueeze(0), x_imag.unsqueeze(0)), 0), 2, dim=0)
        phs = torch.atan2(x_imag.float(), x_real.float()+1e-7).to(x_cuda.dtype)
        if return_acts:
            layer_acts = [x_real, x_imag, mag, phs]

        # Processes Magnitude and phase individually
        mag_hat, m_acts = self.aenc.forward(mag, knobs_cuda, skip_connections='sf', return_acts=return_acts)
        phs_hat, p_acts = self.phs_aenc.forward(phs, knobs_cuda, skip_connections='', return_acts=return_acts)
        if return_acts:
            layer_acts.extend(m_acts)
            layer_acts.extend(p_acts)

        output_phs_dim = phs_hat.size()[1]
        phs_hat = phs_hat + phs[:,-output_phs_dim:,:] # <-- residual skip connection. Slightly smoother convergence

        # Back to Real and Imaginary
        an_real = mag_hat * torch.cos(phs_hat)
        an_imag = mag_hat * torch.sin(phs_hat)

        # Forward synthesis pass
        x_fwdsyn = self.dft_synthesis.forward(an_real, an_imag)

        # final skip residual
        y_hat = x_fwdsyn  + x_cuda[:,-x_fwdsyn.size()[-1]:]/2

        if return_acts:
            layer_acts.extend([mag_hat, phs_hat, an_real, an_imag, x_fwdsyn, y_hat])

        if return_acts:
            return 2*y_hat, mag, mag_hat, layer_acts   # undo the /2 at the beginning
        else:
            return 2*y_hat, mag, mag_hat



class st_model(nn.Module):
    """
    Wrapper routine for AsymMPAEC.  Enables generic call in case we change later
    """
    def __init__(self, scale_factor=1, shrink_factor=4, num_knobs=3, sr=44100, scale_scheme='lean'):
        """
            scale_factor: change dimensionality of run by this factor
            shrink_factor:  output shrink factor, i.e. fraction of output actually trained on

        """
        super(st_model, self).__init__()

        # Data settings
        chunk_size = int(8192 * scale_factor)           # size of audio that NN model expects as input
        out_chunk_size = int(chunk_size/shrink_factor)     # size of output audio that we actually care about

        # save these variables for use when loading models later
        self.scale_factor, self.shrink_factor = scale_factor, shrink_factor
        self.in_chunk_size, self.out_chunk_size = chunk_size, out_chunk_size
        self.num_knobs = num_knobs

        print("Input chunk size =",chunk_size)
        print("Intended Output chunk size =",out_chunk_size)
        print("Sample rate =",sr)

        # Analysis parameters
        ft_size = 1024
        hop_size = 384
        #hop_size = int(256 * scale_factor)

        if scale_scheme != 'lean': # the following doesn't scale well (like O(N^2)), but this is the old scheme, for backwards compatibility
            ft_size = int(ft_size * scale_factor)
            hop_size = int(hop_size * scale_factor)

        expected_time_frames = int(np.ceil(chunk_size/float(hop_size)) + np.ceil(ft_size/float(hop_size)))
        output_time_frames = int(np.ceil(out_chunk_size/float(hop_size)) + np.ceil(ft_size/float(hop_size)))
        y_size = (output_time_frames-1)*hop_size - ft_size
        if y_size != out_chunk_size:
            print(f"Warning: y_size ({y_size}) should equal out_chunk_size ({out_chunk_size})")
            print(f"    Setting out_chunk_size = y_size = {y_size}")
        self.out_chunk_size = y_size
        self.mpaec = AsymMPAEC(expected_time_frames, ft_size=ft_size, hop_size=hop_size, n_knobs=num_knobs, output_tf=output_time_frames)

        #self.freeze()    # TODO: try this out another time

    def clip_grad_norm_(self):
        self.mpaec.clip_grad_norm_()

    def forward(self, x_cuda, knobs_cuda, return_acts=False):
        return self.mpaec.forward(x_cuda, knobs_cuda, return_acts=return_acts)

    '''# not quite ready for this.
    def freeze(self):
        freeze_layers([self.mpaec.dft_analysis, self.mpaec.dft_synthesis])

    def unfreeze(self):
        unfreeze_layers([self.mpaec.dft_analysis, self.mpaec.dft_synthesis])
    '''

# EOF

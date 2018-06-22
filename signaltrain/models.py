
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
    """
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
        ricat_s   = torch.cat((real, imag), 2)                # 'cat' real & imag together; _s for a skip_later
        ricat  = self.act( self.shrink(ricat_s) )        # _s2 for other skip connection  skips=1
        ricat  = self.act( self.shrink2(ricat) )

        # ----- here is the 'middle of the hourglass';  from here we expand

        ricat     = self.act( self.grow2(ricat) )

        ricat     = self.act( self.grow(ricat) )
        if (2 in skips):
            ricat += ricat_s  # skip connection; simple addition uses less memory than map(torch.cat)

        # split the cat-ed real & imag back up [arbitrarily] ;
        uncat = torch.chunk( ricat, 2, dim=2)
        # TODO: actually, this probably sets imag to the value of the skip connection we just made

        y = self.decoder(uncat[0],uncat[1])
        y = self.act(self.back_grow2(y))
        if (3 in skips):
            y += y_s
        y = self.back_grow(y)
        if (4 in skips):  # for final skip we simply add (WaveNet does this). cat is too memory-intensive for large audio clips
            y += input_var

        layers = []# (ricat)     # additional container to report intermediary activations & more
        return y, layers



class FNNManyToOne(nn.Module):
    """
    this seems similar to our above models, and yet it hardly learns at all??

    simple shrinking multi-layer FNN
    this takes an stacked set of sequental windows and predicts a single value (the last one)
    it is a many-to-one model
    """
    def __init__(self, chunk_size=8192, nlayers=13, ratio=2, mode='center'):
        super(FNNManyToOne, self).__init__()
        self.nlayers = nlayers
        self.slicemode = mode
        self.sizes = np.array(chunk_size/np.power(2,range(nlayers))).astype(np.int)
        self.shrinks = nn.ModuleList([nn.Linear( int(self.sizes[l]), int(self.sizes[l+1]), bias=False)
                                 for l in range(nlayers-1)] )
        self.finallayer = nn.Linear(self.sizes[-1],1)
        self.act     = nn.LeakyReLU()

    def forward(self, input_var, skips=(2,3,4)):    # my trials indicate skips=(2,3) works best. (1,2,3) has more noise, (3) converges slower, no skips converges slowest
        y = input_var
        layers = []
        for l in np.arange(self.nlayers-1):
            y = self.act( self.shrinks[l](y) )
            layers.append(y)
        y = self.finallayer(y)
        y += st_utils.slice_prediction(input_var,mode=self.slicemode).unsqueeze(2)  # simple skip connection
        layers.append(y)
        return y, layers



class SpecShrinkGrow_cat3(nn.Module):
    """
    same as above but w/ three front- and back- shrink & grow layers instead 2

    This model first projects the input audio to smaller sizes, using some 'dense' layers,
    Then calls the FNN analysis routine, does show shinking & growing, calls FNNSynthesis,
    and then expands back out to the original audio size.
    Two skip connections are present; more or fewer produced worse results that this.
    For a diagram of thi smodel, see ../docs/model_diagram.png
    """
    def __init__(self, chunk_size=1024):
        super(SpecShrinkGrow_cat, self).__init__()


        ratio = 3   # Reduction ratio. 3 works about as well as 2.  4 seems to be too much
                    # one problem with ratio of 3 is it can produce odd sizes that don't split the data evenly between two GPUs
        size_front_shrink = int(chunk_size/ratio)
        size_front_shrink2 = int(size_front_shrink/ratio)
        ft_size = int(size_front_shrink2/ratio)

        # these shrink & grow routines are just to try to decrease encoder-decoder GPU memory usage
        # and allow for a larger 'reach', temporally
        self.front_shrink  = nn.Linear(chunk_size, size_front_shrink, bias=False)
        self.front_shrink2 = nn.Linear(size_front_shrink, size_front_shrink2, bias=False)
        self.front_shrink3 = nn.Linear(size_front_shrink2, ft_size, bias=False)
        self.back_grow3    = nn.Linear(ft_size, size_front_shrink2, bias=False)
        self.back_grow2    = nn.Linear(size_front_shrink2, size_front_shrink, bias=False)
        self.back_grow     = nn.Linear(size_front_shrink, chunk_size, bias=False)

        # the "FFT" routines
        self.encoder = front_end.FNNAnalysis(ft_size=ft_size)   # gives matrix mult. size mismatches
        self.decoder = front_end.FNNSynthesis(ft_size=ft_size)#, random_init=True)  #  random_init=True gives me better Val scores

        ft_out_dim  = int(ft_size/2+1)   # this is the size of the output from the FNNAnalysis routine
        full_dim = 2*ft_out_dim          # we will cat the real and imag halves together
        #self.cat_ri  = nn.Linear(2*full_dim,  full_dim, bias=False)

        # define some more size variables for shrinking & growing sizes in between encoder & decoder
        med_dim   = nearest_even(full_dim / ratio)
        small_dim = nearest_even(med_dim / ratio)


        self.shrink  = nn.Linear(full_dim,  med_dim, bias=False)
        self.shrink2 = nn.Linear(med_dim,   small_dim, bias=False)
        #self.dense = nn.Linear(self.small_dim, self.small_dim, bias=False)  # not needed
        self.grow2   = nn.Linear(small_dim, med_dim, bias=False)
        self.grow    = nn.Linear(med_dim,   full_dim, bias=False)

        self.mapskip  = nn.Linear(2*full_dim,  full_dim, bias=False)  # maps concatenated skip connection to 'regular' size
        self.mapskip2 = nn.Linear(2*med_dim,   med_dim, bias=False)
        self.mapbigskip = nn.Linear(2*size_front_shrink2,   size_front_shrink2, bias=False)
        #self.finalskip = nn.Linear(2*chunk_size,   chunk_size, bias=False)  # does not fit in CUDA memory

        self.act     = nn.LeakyReLU()   # Tried ReLU, ELU, SELU; Leaky seems to work best.  SELU yields instability

    def forward(self, input_var, skips=(2,3,4)):    # my trials indicate skips=(2,3) works best. (1,2,3) has more noise, (3) converges slower, no skips converges slowest
        y_orig = input_var
        y = self.act(self.front_shrink(y_orig))
        y_s = self.act( self.front_shrink2(y) )   # _s to prepare for skip connection "skips=3"
        y = self.act( self.front_shrink3(y_s) )
        real, imag = self.encoder(y)
        ricat_s   = torch.cat((real, imag), 2)                # 'cat' real & imag together; _s for a skip_later
        ricat_s2  = self.act( self.shrink(ricat_s) )        # _s2 for other skip connection  skips=1
        ricat     = self.act( self.shrink2(ricat_s2) )

        # ----- here is the 'middle of the hourglass';  from here we expand

        ricat     = self.act( self.grow2(ricat) )
        if (1 in skips):  # this one has no discernable effect; but does yield more noise
            ricat = self.mapskip2(torch.cat((ricat, ricat_s2), 2)) # make skip connection

        ricat     = self.act( self.grow(ricat) )
        if (2 in skips):
            ricat = self.mapskip(torch.cat((ricat, ricat_s), 2))   # make skip connection

        uncat = torch.chunk( ricat, 2, dim=2)       # split the cat-ed real & imag back up [arbitrarily] ;

        # TODO: actually, this next line probably sets imag to the value of the skip connection we just made
        real, imag = uncat[0],uncat[1]
        y = self.decoder(real, imag)
        y = self.act(self.back_grow3(y))
        if (3 in skips):
            y = self.mapbigskip( torch.cat((y, y_s), 2)  )
        y = self.back_grow2(y)
        y = self.back_grow(y)
        #if (4 in skips):
        #    y = self.finalskip( torch.cat((y, y_orig), 1) )  # too memory-intensive for large chunk_size

        layers = (ricat)     # additional container to report intermediary activations & more
        return y, layers


class SpecShrinkGrow(nn.Module):
    """
    'original' spectral bottleneck encoder

    This model first projects the input audio to smaller sizes, using some 'dense' layers,
    Then calls the FNN analysis routine, does show shinking & growing, calls FNNSynthesis,
    and then expands back out to the original audio size.
    Two skip connections are present; more or fewer produced worse results that this.
    For a diagram of thi smodel, see ../docs/model_diagram.png
    """
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

        print("chunk_size, mid_size, ft_size = ",chunk_size, mid_size, ft_size)
        print("full_dim, med_dim, small_dim = ",full_dim, med_dim, small_dim)

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
        y = self.act(self.back_grow2(y))
        if (3 in skips):
            y = self.bigskip( torch.cat((y, y_s), 2) )
        y = self.back_grow(y)
        #if (4 in skips):
        #    y = self.finalskip( torch.cat((y, y_orig), 1) )  # too memory-intensive for large audio clips

        layers = ()     # additional container variable that could be used to report intermediate values
        return y, layers




class SpecFrontBack(nn.Module):
    """
    This calls SM's front_end routines with no temporal pre-processing

    For the compressor effect, this results in output that does not match the
    target output, rather the ouput looks like the input signal reduced in
    amplitude so that it 'splits the difference' between input & target
    """
    def __init__(self, chunk_size=1024):
        # Note: Maximum chunk_size for this model is around 8192 on Titan X GPU
        super(SpecFrontBack, self).__init__()

        # the "FFT" routines
        self.encoder = front_end.FNNAnalysis(ft_size=chunk_size)   # gives matrix mult. size mismatches
        self.decoder = front_end.FNNSynthesis(ft_size=chunk_size)#, random_init=True)  #  random_init=True gives me better Val scores

        full_dim = (chunk_size+2)   # full size of 'catted' real & imag parts
        ratio = 3                   # factor of 3 works well, 4 is too much
        med_dim = int(full_dim/ratio)
        small_dim = int(med_dim/ratio)

        self.shrink  = nn.Linear(full_dim,  med_dim, bias=False)
        self.shrink2  = nn.Linear(med_dim,  small_dim, bias=False)
        self.grow2  = nn.Linear( small_dim, med_dim, bias=False)    # matches shrink2
        self.grow  = nn.Linear( med_dim, full_dim, bias=False)      # matches shrink

        self.mapskip = nn.Linear( chunk_size*2, chunk_size, bias=False)  # when we cat a skip connection in, we need to reduce to 'normal size'

        # a nonlinear activation; # Tried ReLU, ELU, SELU; Leaky seems to work best.  SELU yields instability
        self.act     = nn.LeakyReLU(negative_slope=0.1)    # changing negative slope from 0.01 to 0.1 has almost no effect

    def forward(self, input_var):
        real1, imag1 = self.encoder(input_var)      # transform from time domain to spectral domain

        # put real & imag together, and add intermediate layers
        ricat = torch.cat((real1, imag1), 2)            # "ricat" = "real/imaginary cat"
        #print("ricat.size() = ",ricat.size())
        ricat_act = self.act(ricat)                     # nonlinear activation

        # intermediary "hourglass" or bottleneck  - including this part improves learning significantly
        hg_s = self.act( self.shrink(ricat_act) )
        hg_s2 = self.act( self.shrink2(hg_s) )
        hg_g2 = self.act( self.grow2(hg_s2) )
        hg_g = self.act( self.grow(hg_g2) )         # we are now back up to full size

        # now run the decoder/synthesis
        uncat = torch.chunk( hg_g, 2, dim=2)       # split the cat-ed real & imag back up [arbitrarily] ;
        real2, imag2 = uncat[0],uncat[1]
        synthed = self.decoder(real2, imag2)

        # skip connection helps learning but is a memory hog for large chunk_size
        #    This is why we prefer models that shrink in the temporal domain first,
        #    i.e. to shrink the size of the model
        catted = torch.cat((synthed, input_var), 2)
        catted_act = catted # self.act(catted)   # adding activation has almost no effect, might hurt
        output_var = self.mapskip(catted_act)
        #output_var = self.act(output_var)   # don't do this. ruins training; makes output end up as nearly zero

        layers = (ricat_act, hg_s, hg_g, synthed)                  # report additional intermediary values
        return output_var, layers




class Seq2Seq(nn.Module):
    """
    SHH's attempt at a Seq2Seq model.

    TODO: Fix problem: Runs of memory.
    """
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


class One_Hot(nn.Module):
    """
    One hot encoding, used by WaveNet (below)

    From  https://gist.github.com/lirnli/4282fcdfb383bb160cacf41d8c783c70
    """
    def __init__(self, depth, device="cpu"):   # depth = mu of mu-law companding
        super(One_Hot,self).__init__()
        self.depth = depth
        self.ones = torch.torch.eye(depth).to(device)
    def forward(self, X_in):
        return torch.index_select(self.ones, 0, X_in.data)
    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

class WaveNet(nn.Module):
    """
    Google's (van den Oord et al's) WaveNet model, implemention from
    https://gist.github.com/lirnli/4282fcdfb383bb160cacf41d8c783c70

    This assumes that the input and target signals have been mu-law companded
    """
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

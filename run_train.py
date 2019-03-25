#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Scott H. Hawley'
__version__ = '0.0.2'

# imports
import numpy as np
import torch
import os
import sys
import glob
import argparse
import matplotlib
matplotlib.use('Agg')
import signaltrain as st

if __name__ == "__main__":

    # Set up random number generators and decide which device we'll run on
    np.random.seed(218)
    torch.manual_seed(218)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(218)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type('torch.FloatTensor')


    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Trains neural network to reproduce input-output transformations.",\
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch', type=int, help="batch size", default=200)
    parser.add_argument('--effect', help='Name of effect to use. ("files" = search for "target_" and effect_info.ini files in path)', default="comp_4c")
    parser.add_argument('--epochs', type=int, help='Number of epochs to run', default=1000)
    parser.add_argument('--path', help='Directory to pull input (and maybe target) data from (default: None, means only synthesized-on-the-fly data)', default=None)
    parser.add_argument('-n', '--num', type=int, help='Number of "data points" (audio clips) per epoch', default=200000)
    parser.add_argument('--sr', type=int, help='Sampling rate', default=44100)
    parser.add_argument('--scale', type=float, help='Scale factor (of input size & whole model)', default=1.0)
    parser.add_argument('--shrink', type=int, help='Shink output chunk relative to input by this divisor', default=4)
    parser.add_argument('--apex', help="optimization setting to use with NVIDIA apex", default="O0")
    args = parser.parse_args()

    # print command line as it was invoked (for reading nohup.out later)
    print("Command line: "," ".join(sys.argv[:]))

    # Check arguments before beginning to train....

    # establish which audio effect class is being used
    e = args.effect
    if e == 'files':   # target outputs are given as files rather than 'live' 'plugins'
        # TODO: check to make sure there are a suitable number of 'target' files in path
        effect = st.audio.FileEffect(args.path)
    elif e == 'comp_4c':
        effect = st.audio.Compressor_4c()
    elif e == 'comp':
        effect = st.audio.Compressor()
    elif e == 'comp_t':
        effect = st.audio.Comp_Just_Thresh()
    elif e == 'denoise':
        effect = st.audio.Denoise()
    elif e == 'lowpass':
        effect = st.audio.LowPass()
    elif 'VST' in e:
        print("VST plugins not integrated yet, but that would be great.")
        print("Feel free to grab Igor Gadelha' VSTRender lib to help implement this.")
        print("See https://github.com/igorgad/dpm")
        sys.exit(1)
    else:
        print(f"Effect option '{e}' is not yet added")
        sys.exit(1)

    # this is just to avoid confusion: the datagenerator class will/should trap for this also.
    if (args.path is None) or (not glob.glob(args.path+"/Train/input*")) \
        or (not glob.glob(args.path+"/Val/input*")):  # no input files = 100% probability of synth'ing input data
        args.synthprob = 1.0    # this isn't used yet, but passed anyway for later use
    if effect is st.audio.FileEffect:
        args.synthprob = 0.0    # can't run pre-recorded effects post-facto

    # Finished parsing/checking arguments, ready to run


    st.misc.print_choochoo(__version__)  #  ascii art is the hallmark of professionalism

    print("Running with args =",args)

    # call the trianing routine
    st.train.train(epochs=args.epochs, n_data_points=args.num, batch_size=args.batch, device=device, sr=args.sr,\
        effect=effect, datapath=args.path, scale_factor=args.scale, shrink_factor=args.shrink,
        apex_opt=args.apex)

# EOF

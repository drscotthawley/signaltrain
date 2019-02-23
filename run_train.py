#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Scott H. Hawley'
__version__ = '0.0.2'

# imports
import numpy as np
import torch
import os, sys
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


    st.misc.print_choochoo(__version__)             #  ascii art makes ppl smile


    # parse command line arguments
    parser = argparse.ArgumentParser(description="Trains neural network to reproduce input-output transformations.")
    parser.add_argument('-b', '--batch', type=int, help="batch size", default=200)
    parser.add_argument('--effect', help='Name of effect to use', default="comp_4c")
    parser.add_argument('--epochs', type=int, help='Number of epochs to run', default=1000)
    parser.add_argument('--path', help='Directory to pull data from (None for synthesized data)', default=None)
    parser.add_argument('-n', '--num', type=int, help='Number of "data points" (audio clips) per epoch', default=200000)
    parser.add_argument('--sr', type=int, help='Sampling rate', default=44100)
    args = parser.parse_args()
    print("Running with args =",args)


    # establish which audio effect class is being used
    if args.path is not None:
        effect = st.audio.FileEffect(args.path)
    elif args.effect == 'comp_4c':
        effect = st.audio.Compressor_4c()
    else:
        print("That effect option is not yet added")
        sys.exit(1)


    # call the trianing routine
    st.train.train(epochs=args.epochs, n_data_points=args.num, batch_size=args.batch, device=device,\
        effect=effect, datapath=args.path)

# EOF

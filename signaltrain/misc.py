# -*- coding: utf-8 -*-
__author__ = 'S.H. Hawley'
import torch
import os
import sys
import numpy as np


def print_choochoo(__version__):
    """Just for fun. Makes a train picture."""
    print(" ~.~.~.~.      ")
    print(" ____    `.    ")
    print(" ]DD|_n_n_][   ")
    print(" |__|_______)  ")
    print(" 'oo OOOO oo\_ ")
    print("~+~+~+~+~+~+~+~")
    print("SignalTrain "+__version__)
    print("")


def save_checkpoint(checkpointname, model, epoch, parallel, optimizer, effect, sr):
    """
    Saves a dictionary to a tar file. Package it with model state_dict and run parameters
    """
    print(f'\nsaving model to {checkpointname}',end="")
    model2save = model.module  if parallel else model
    state_dict = model2save.state_dict()
    state = {'epoch': epoch + 1, 'state_dict':  state_dict,
        'optimizer': optimizer.state_dict(),
        'effect_name': effect.name,
        'knob_names': effect.knob_names, 'knob_ranges': effect.knob_ranges,
        'scale_factor': model2save.scale_factor, 'shrink_factor': model2save.shrink_factor,
        'in_chunk_size': model2save.in_chunk_size,'out_chunk_size': model2save.out_chunk_size,
        'sr': sr}
    torch.save(state, checkpointname)


def load_checkpoint(checkpointname, fatal=False):
    """
    load up the checkpoint if it exists
    for backwards-compatibility: guess some common run parameters if they're not in the checkpoint
    """
    state_dict, rv = {}, {}    # rv = run_values

    if os.path.isfile(checkpointname):
        print("\n***** Checkpoint file found. Loading weights.")
        checkpoint = torch.load(checkpointname) # map_location=device)
        state_dict = checkpoint['state_dict']

        # Guess some typical run values in case they're not in the checkpoint file
        rv.setdefault('sr', 44100)
        rv.setdefault('scale_factor', 1)
        rv.setdefault('shrink_factor', 4)
        rv.setdefault('in_chunk_size', 8192)
        rv.setdefault('out_chunk_size', 2048)
        rv.setdefault('knob_names', ['thresh', 'ratio', 'attackTime','releaseTime'])
        rv.setdefault('knob_ranges', np.array([[-30,0], [1,5], [1e-3,4e-2], [1e-3,4e-2]]) )

        for key, value in checkpoint.items():
            if 'state_dict' not in key:  # everything that's not state_dict becomes a run value
                rv[key] = value
    elif fatal:
        print("Error, no checkpoint found")
        sys.exit(1)

    return state_dict, rv

#! /usr/bin/env python3
"""
Convert a checkpoint file to another file, e.g. one for deployment only (no optimizer info)

Intended as a general utility that might one day convert to ONNX or Tensorflow
formats, but currently ONNX doesn't support all the operations we need (e.g.,
flip() and atan2(). )

Author: Scott H. Hawley 
"""

import numpy as np
import torch
import os, sys
sys.path.append(os.path.abspath('../'))  # for running from signaltrain/demo/
import signaltrain as st
from signaltrain.nn_modules import nn_proc
import argparse
import faulthandler
faulthandler.enable()


def save_model_deploy(tarfilename, model, knob_names, knob_ranges, sr):
    """
    Like saving a checkpoint, but without optimizer or epoch info
    """
    print(f'\nSaving model to {tarfilename}')
    state_dict = model.state_dict()
    state = {
       'state_dict':  state_dict,
        'knob_names': knob_names, 'knob_ranges': knob_ranges,
        'scale_factor': model.scale_factor, 'shrink_factor': model.shrink_factor,
        'in_chunk_size': model.in_chunk_size,'out_chunk_size': model.out_chunk_size,
        'sr': sr
        }
    torch.save(state, tarfilename)


# parse command line args
parser = argparse.ArgumentParser(description="Converts model another format",\
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('in_name', help='Name of the checkpoint file to read', default="modelcheckpoint.tar")
parser.add_argument('out_name', help='Name of the output file to write', default="modelcheckpoint_deploy.tar")
parser.add_argument('--format', help='format to convert to', choices=['deploy'], default="deploy")
args = parser.parse_args()

# read the checkpoint file
state_dict, return_values = st.misc.load_checkpoint(args.in_name, device="cpu", fatal=True)
print("Model successfully loaded")

# parse the checkpoint file
rv = return_values   # just for brevity
scale_factor = rv['scale_factor']
shrink_factor = rv['shrink_factor']
knob_names = rv['knob_names']
knob_ranges = rv['knob_ranges']
num_knobs = len(knob_names)
sr = rv['sr']

print("Model info from checkpoint:")
print(" scale_factor, shrink_factor, knob_names, knob_ranges, num_knobs, sr =",
    scale_factor, shrink_factor, knob_names, knob_ranges, num_knobs, sr)
print("")


# define the model
model = nn_proc.st_model(scale_factor=scale_factor, shrink_factor=shrink_factor, num_knobs=num_knobs, sr=sr)
model.load_state_dict(state_dict)  # overwrite weights using checkpoint info

# write the new outputfile
save_model_deploy(args.out_name, model, knob_names, knob_ranges, sr)

print("Finished successfully.")

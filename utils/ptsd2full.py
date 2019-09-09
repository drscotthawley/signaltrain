#! /usr/bin/env python3

'''
Converts a PyTorch state-dict-only model file to a "full" model
See  https://pytorch.org/tutorials/beginner/saving_loading_models.html

Note that YOU have to edit this file -- including adding imports to tell it where your
model source code is.

Also note that full pytorch models are local-path-dependent, so they break if you move them elsewhere
'''

import torch
import argparse
import os, sys
import inspect
from pathlib import Path


# Here's where you put in your own model definition for TheModelClass
sys.path.append('../signaltrain')
import nn_proc
TheModelClass = nn_proc.st_model


def load_model(infile):
    if not os.path.isfile(infile):
        print(f"Error: file {infile} not found.")
        sys.exit(1)


    # read the checkpoint file, first to cpu
    checkpoint = torch.load(infile, map_location='cpu')
    print("checkpoint.keys() = ",checkpoint.keys())


    # grab any model-setup-info parameters needed from the checkpoint
    model_keys = inspect.getargspec(TheModelClass)
    print("TheModelClass expects kwargs:", model_keys)

    args, kwargs = [], {}
    # populate kwargs with anything the model needs that is given in cp_keys
    for key, value in checkpoint.items():
        print(" checking key = ",key)
        if key in model_keys.args:
            print(f"    **** Hey: key {key} is in both")
            kwargs[key] = value
            print(f"          setting kwargs[{key}] = {value}")

    # set any more 'custom' kwargs you need
    kwargs['num_knobs'] = len(checkpoint['knob_names'])  # my special thing

    # setup the model  # <--- You have to do this yourself **
    model = TheModelClass(*args, **kwargs)


    # populate the model weights from the state dict
    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint.items()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Converts a PyTorch state-dict-only model file to a "full" model',\
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile', help='Input Name of state-dict PyTorch file', default="../demo/modelcheckpoint.tar")
    parser_args = parser.parse_args()

    # check that infile exists
    infile = parser_args.infile

    model, _ = load_model(infile)

    # new filename
    outfile = Path(infile).stem + ".ptf"

    # save full model
    torch.save(model, outfile)

# EOF

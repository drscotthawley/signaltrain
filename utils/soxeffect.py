#! /usr/bin/env python

"""
This will run a sox effect using a range of control parameters on a given list
of input files.  For a singe file, with no range to the control parameters,
running this script is EQUIVALENT to running sox itself.

Where this script differs is that commas separating pairs of numbers will be
regarded as defining min & max values for a range of a given parameter value

Usage:
  ./soxeffect.py <effect_name> '<settings_string>' <input_files>

e.g., to run a chorus effect with only one setting
./soxeffect.py chorus '0.7 0.9 55.0 0.4 0.25 2.0 -s' input*.wav

Alternative: supply a range  of settings, with commas separating min/max values:
./soxeffect.py chorus '0.7,0.9 0.5,0.9 40.0,60.0 0.2,0.5 0.10,.4 1.0,3.0 -s' input*.wav

Results:
  A set of output files, one per input file, names in which 'target' is
  prepended and 'input removed' along with effect settings.
"""

import numpy as np
import torch
import sys
import os
sys.path.append('/home/shawley/signaltrain')
sys.path.append('..')
import signaltrain as st
import glob
import argparse
import re


def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def ranges_to_vals(str):
    """replace the first instance of a comma-separated pair of max,min numbers
       with a value randomly generated in between them
    """
    out_str = ''                # this will be the final output
    pvals, pranges = [], []     # list of parameter values used for naming files
    pars = str.split(' ')       # first split on spaces
    for p in pars:
        mm = p.split(',')  # split at comma for min & max
        valstr = mm[0]
        if is_number(valstr) and len(mm) > 1: # if we have a range of numbers
            minval, maxval = float(valstr), float(mm[1])
            pranges.append([minval, maxval])
            val = minval +  np.random.rand()*(maxval - minval)
            valstr = f'{val:.3f}'           # trunacte to 3 decimal points
            pvals.append(valstr)
        out_str += f'{valstr} '
    return out_str, pvals, pranges

def test_ranges_to_vals():   # just a sample test
    ranges_to_vals('0.7,.8 0.9 55.0 0.4 0.25,.3 2.0 -s')



if __name__ == "__main__":
    # parse command line args
    parser = argparse.ArgumentParser(description="Applies a sox audio effect to lots of files",\
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('effect', help='Name of effect class for generating target')
    parser.add_argument('params', help='String of control settings')
    parser.add_argument('inputs', nargs='+', help='List of input files')

    args = parser.parse_args()
    print("args =",args)
    n_ranges = args.params.count(',')   # How many paramter ranges were specified

    # if there are no ranges to the effects, then just call sox on each file
    for in_file in args.inputs:
        paramstr, pvals, pranges = ranges_to_vals(args.params)       # replace ranges with random #s
        out_file = 'target'+in_file.replace('input','')     # remove input & prepent target
        if len(pvals) > 0:
            pvalstr = ''
            for p in pvals:    # note that pvals are strings, not numbers
                pvalstr += f'__{p}'
        else:
            pvalstr = '_1'
        out_file = out_file.replace('_.wav',f'{pvalstr}.wav')
        execstr = f'sox --multi-threaded {in_file} {out_file} {args.effect} {paramstr}'
        print("  execstr = ",execstr)
        os.system(execstr)              # Execute sox for this setting

print("\n\nCopy & paste the following to use as effect.ini file:\n")
if n_ranges > 1:
    knob_names = [f'p{n}' for n in range(n_ranges)]
    knob_ranges = pranges
else:
    knob_names = "['p1']"
    knob_ranges = "[.999, 1.001]"

print(f"""[effect]
name = sox_{args.effect}
knob_names = {knob_names}
knob_ranges = {knob_ranges}""")

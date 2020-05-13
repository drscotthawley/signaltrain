#! /usr/bin/env python3
# ^ Note: this code uses f-strings, which require python 3.6 and higher

# Purpose:
#  This checks for inconsistencies between input-output pairs in datasets (in bulk)
#
#  Author:
#   Scott H. Hawley (scott.hawley@belmont.edu),  May 12, 2020
#
# Methodology:
#   It checks the lengths of the audio, and then performs a cross correlation
#   to measure timing 'skew' or offset.
#
# Usage:
#    $ ./check_timing.py [options] <input> <target>
#  or
#    $ ./check_timing.py [options] <directory>
#  or
#    $ ./check_timing.py [options] <input1> <input2>... <target1> <target2>...
##
#    If <directory> is specified, it will search for an 'input' file
#    and match it with a corresponding 'target' file with the name number, e.g.
#    "input_47_" and "target_47_".
#    If there are subdirectories in <directory> (e.g. Train/ and Test/), it will
#    descend into those.  Watch out if you have symbolic directory links inside
#     your main dataset.
#
# Command-line options:
#    By default it simply outputs analysis data, and takes no actions to fix anything.
#    The following "fix" options will all make changes to the dataset *in place*.
#     meaning the existing dataset is *overwritten*. Thus it is recommended that
#     you only run this script on a *copy* of your dataset,
#    Run the script first without them to see what it's going to do.

#  -a    (Time-)Align the audio, using cross-correlation.  (Don't use this if
#            you've got an 'echo' effect!)
#  -d    Delete any extra files, i.e. input files witout corresponding target outputs,
#            or vice versa.
#  -l    Fix the length: truncate any extra audio appearing in one file but not
#            the other.  (Runs after -a)
#  -m    Force mono (who knows, maybe somebody made a stereo file?)
#  -s    Sample rate; in this case it will force all the files have the same sample rate
#          as that of the first input file it encounters (whatever that is)
#  --fix:  All of the above, i.e. "--fix" == "-adlms"
#    Again, you may not want all the --fix options, so...use with care.
#
#  Example:
#   $ ./check_dataset.py datasets/SignalTrain_LA2A_dataset_rev1/Train/*.wav
# or
#   $ ./check_dataset LA2A_Dataset/           # to check
#  then
#   $ ./check_dataset --fix LA2A_Dataset/     # to fix everything

from scipy import signal
import numpy as np
import argparse
import sys
import os
import glob
import librosa
from scipy.io import wavfile
import shutil
import re

DEBUG = False

if DEBUG:
    import matplotlib
    matplotlib.use('TkAgg')   # use a raster backend for plotting many points
    import matplotlib.pyplot as plt


class colors():    # Because I'm lazy
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


def estimate_time_shift(x, y, sr = 44100):
    """ Computes the cross-correlation between time series x and y, grabs the
        index of where it's a maximum.  This yields the time difference in
        samples between x and y.
    """
    if DEBUG: print("computing cross-correlation")
    corr = signal.correlate(y, x, mode='same', method='fft')
    if DEBUG: print("finished computing cross-correlation")

    nx, ny = len(x), len(y)
    t_samples = np.arange(nx)
    ct_samples = t_samples - nx//2  # try to center time shift (x axis) on zero
    cmax_ind = np.argmax(corr)      # where is the max of the cross-correlation?
    dt = ct_samples[cmax_ind]       # grab the time shift value corresponding to the max c-corr
    if DEBUG: print("cmax_ind, nx//2, ny//2, dt =",cmax_ind, nx//2, ny//2, dt)

    if DEBUG:
        fig, (ax_x, ax_y, ax_corr) = plt.subplots(3, 1)
        ax_x.get_shared_x_axes().join(ax_x, ax_y)
        ax_x.plot(t_samples, x)
        ax_y.plot(t_samples, y)
        ax_corr.plot(ct_samples, corr)
        plt.show()
    return dt


#  for use in filtering filenames
def is_acceptable(filename):
    return filename.lower().endswith(('.wav', '.mp3', '.aif', '.aiff')) and \
        (('input_' in filename) or ('target_' in filename))



parser = argparse.ArgumentParser(description="Check dataset for mismatches",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input_or_dir', help='input file 1, or directory')
parser.add_argument('target_or_more_files', nargs='*', help='target file 1, or optional more files (for non-directory usage)')
parser.add_argument('-a','--align', help='Fix: Align time (overwrites)', action='store_true')
parser.add_argument('-d','--delete', help='Fix: Delete extra/unmatching input or target files (overwrites)', action='store_true')
parser.add_argument('-f','--fast', help='Fast: skip timing checks', action='store_true')
parser.add_argument('-l','--length', help='Fix: Make lengths the same, by truncating (overwrites)', action='store_true')
parser.add_argument('-m','--mono', help='Fix: Force mono (overwrites)', action='store_true')
parser.add_argument('-s','--sr', help='Fix: Enforce sample rate of first input (overwrites)', action='store_true')
parser.add_argument('--fix', help='Fix: Apply all fixes (overwrites)', action='store_true')
args = parser.parse_args()
if (args.fix):
    [args.align, args.length, args.delete, args.sr, args.mono] = [True]*5
if DEBUG: print("args =",args)

# Make sense of how the user is specifying where to check
if args.target_or_more_files == []:
    dir = args.input_or_dir
    assert os.path.isdir(dir), f"{dir} is not a directory"

    print(f"Operating on directory {dir}")
    file_list, input_list, target_list = [], [], []
    # TODO: make sure it's actually a directory

    for dirpath, subdirs, files in os.walk(dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3', '.aif', '.aiff')):
                if 'input' in f:
                    input_list.append(os.path.join(dirpath, f))
                elif 'target' in f:
                    target_list.append(os.path.join(dirpath, f))
            if is_acceptable(f):
                file_list.append(os.path.join(dirpath, f))
else:
    file_list = [args.input_or_dir] + args.target_or_more_files
    print(f"Operating on a list of {len(file_list)} files")
    # make a list of all the inputs, and a list of all the tagets
    input_list = list(filter(lambda x: 'input' in x, file_list))
    target_list = list(filter(lambda x: 'target' in x, file_list))

input_list.sort()
target_list.sort()


print("\n#### SIMPLE SANITY CHECKS based on filenames. Fast")

# sanity check: as many inputs as targets?
#   Note: one could imagine multiple targets for the same input, but we've
#   not done that for signaltrain.
ni, nt = len(input_list), len(target_list)

# TODO: make it tell us specifically what's lacking or extra
if ni != nt:
    print(f"{colors.RED}**PROBLEM**:{colors.RESET} {ni} inputs but {nt} targets")
    input_nums = [re.search('_[0-9]+_', os.path.basename(i)).group() for i in input_list]
    target_nums = [re.search('_[0-9]+_', os.path.basename(i)).group() for i in target_list]

    for i in input_nums:  # TODO: slow. make this faster with pythonic list operations
        if not (i in target_nums):
            print(f'  {i} is in inputs but not targets')
    for i in target_nums:
        if not (i in input_nums):
            print(f'  {i} is in targets but not inputs')
    sys.exit(1)

# total list of files
file_list = input_list + target_list

# Show what we'll be checking
if DEBUG: print("file_list = ",file_list)


# make sure same file doesn't exist in multiple directories
basenames = [os.path.basename(p) for p in file_list]   # grab all the filenames
assert len(basenames) == len(set(basenames)), "You've got duplicates"

# Loop through files
for i in range(ni):
    problem = False
    input_filename, target_filename = input_list[i], target_list[i]
    ibase, tbase = os.path.basename(input_filename), os.path.basename(target_filename)
    #print(f"input = {input_filename},    target = {target_filename}")

    # make sure the first is an input and the second is a target
    assert ('input_' in ibase) and ('target_' in tbase)

    # make sure the number-designation (first numbers found) of the files line up
    input_num = re.search('_[0-9]+_', ibase).group()
    target_num = re.search('_[0-9]+_', tbase).group()
    if input_num != target_num:
        print(f"{colors.RED}    **PROBLEM**:{colors.RESET} For input = {input_filename},  target = {target_filename}:")
        print(f"                 input_num ({input_num}) != target_num ({target_num})")
        sys.exit(1)
    # make sure they're in the same directory
    assert os.path.dirname(input_filename) == os.path.dirname(target_filename)



print("#### CHECKING THE AUDIO.  Slower.")
# Loop through files
for i in range(ni):
    problem = False
    input_filename, target_filename = input_list[i], target_list[i]
    ibase, tbase = os.path.basename(input_filename), os.path.basename(target_filename)
    print(f"input = {input_filename},    target = {target_filename}")


    repaired = False     # flag for if we want to output a fixed set of files

    # Read the audio files.  x, y = data for input, target
    x, sr_x = librosa.load(input_filename, sr=None, mono=False)
    y, sr_y = librosa.load(target_filename, sr=None, mono=False)

    # Check basic stuff
    if sr_x != sr_y:
        print(f"{colors.RED}    **PROBLEM**:sr_x ({sr_x}) != sr_y ({sr_y}){colors.RESET}")
        if args.sr:
            sr_y, repaired = sr_x, True
            print("     Fixing: setting sr_y := sr_x")
        else:
            problem = True

    if x.shape != y.shape:
        print(f"{colors.RED}    **PROBLEM**: x.shape ({x.shape}) != y.shape ({y.shape}){colors.RESET}")
        problem = True

    if args.mono:
        if len(x.shape) > 1:
            x = x[0,:]
            repaired = True
        if len(y.shape) > 1:
            y = y[0,:]
            repaired = True


    ### Check timing alignment.  Slow
    if not args.fast:
        #Compute the time delay (argmax of cross-correlation) in samples
        if DEBUG: print("    Calling estimate_time_shift (slow)")
        nx = len(x)
        short_len = nx//10       # 20-minute long audio files take a while, so use a subset
        dt = estimate_time_shift(x[0:short_len], y[0:short_len])
        if dt != 0:
            print(f"{colors.RED}    **PROBLEM**: Estimated time shift of {dt} samples from input to target.{colors.RESET}")
            problem = True

            if args.align:     # Fix the alignment
                print("        Trying to fix alignment...")
                if dt < 0:
                    x = x[-dt:]
                else:
                    y = y[dt:]
                newlen = min(x.shape[0], y.shape[0])
                x, y = x[0:newlen], y[0:newlen]

                # check to see how we did
                dt  = estimate_time_shift(x[0:short_len], y[0:short_len])
                print(f"        New estimated time shift = {dt} samples, x.shape = {x.shape}, y.shape = {y.shape}")
                if dt == 0:
                    problem, repaired = False, True
                else:
                    assert False, "Can't figure out what to do with this."

    if not problem:
        print(f" {colors.GREEN}  Looks good! :-) {colors.RESET}")

    if repaired: # save -- overwrite -- new versions of input & output
        print("       Overwriting new version of input and target...")
        wavfile.write(input_filename, sr_x, x)
        wavfile.write(target_filename, sr_y, y)

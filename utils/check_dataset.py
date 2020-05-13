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
#
# (there's no option where it takes a list of files longer than 2, not 'even' an
# even-numbered list, because I just imagined problems with file ordering from
# '*' operators, in cases there are inputs without matching targets, etc. )
#  Also, if you've got a TON of files, the '*' operator might give you a "List
#  too long" error, and then the code might run strangely.)
#
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


parser = argparse.ArgumentParser(description="Check dataset for mismatches",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input_or_dir', help='input file 1, or directory')
parser.add_argument('target_or_more_files', nargs='*', help='target file 1, or optional more files (for non-directory usage)')
parser.add_argument('-a','--align', help='Fix: Align time (overwrites)', action='store_true')
parser.add_argument('-d','--delete', help='Fix: Delete extra/unmatching input or target files (overwrites)', action='store_true')
parser.add_argument('-l','--length', help='Fix: Make lengths the same, by truncating (overwrites)', action='store_true')
parser.add_argument('-m','--mono', help='Fix: Force mono (overwrites)', action='store_true')
parser.add_argument('-s','--sr', help='Fix: Enforce sample rate of first input (overwrites)', action='store_true')
parser.add_argument('--fix', help='Fix: Apply all fixes (overwrites)', action='store_true')
args = parser.parse_args()
if (args.fix):
    [args.align, args.length, args.delete, args.sr, args.mono] = [True]*5
print("args =",args)

# Make sense of how the user is specifying where to check
if args.target_or_more_files == None:
    print("Operating on a directory")
    assert False, "***Actually this doesn't work yet.  Use the list-of-files approach"
    #file_list = glob.glob(args.input_or_dir+'/input*') + glob.glob(args.input_or_dir+'/target*')
    file_list = []
    # TODO: make sure it's actually a directory

    for root, dirs, files in os.walk(args.input_or_dir):
        # TODO: Make sure we only add "input" or "target" files and ignore others!
        file_list.append(files)
else:
    file_list = [args.input_or_dir] + args.target_or_more_files
    print(f"Operating on a list of {len(file_list)} files")

file_list.sort()

# Show what we'll be checking
print("file_list = ",file_list)

# Simple check to make sure we've got an even number of files
n = len(file_list )
assert n%2 == 0, f"ERROR: len(file_list) = {n}, needs to be an even number"

# At this point, we should have a list consisting of
#        [input1, input2, input3,...target1, target2, target3]
for i in range(n//2):
    problem = False
    input_filename, target_filename = file_list[i], file_list[i+n//2]
    ibase, tbase = os.path.basename(input_filename), os.path.basename(target_filename)
    print(f"input = {ibase},    target = {tbase}")

    #### SIMPLE SANITY CHECKS

    # make sure the first is an input and the second is a target
    assert ('input_' in ibase) and ('target_' in tbase)

    # make sure the number-designation (first numbers found) of the files line up
    input_num = re.search('_[0-9]+_', ibase).group()
    target_num = re.search('_[0-9]+_', tbase).group()
    assert input_num == target_num, f"input_num ({input_num}) != target_num = ({target_num})"


    #### CHECKING THE AUDIO

    # Read the audio files
    #  x, y = data for input, target
    x, sr_x = librosa.core.load(input_filename, sr=None, mono=False)
    y, sr_y = librosa.core.load(target_filename, sr=None, mono=False)

    # Check basic stuff
    if sr_x != sr_y:
        print(f"{colors.RED}    **PROBLEM**:sr_x ({sr_x}) != sr_y ({sr_y}){colors.RESET}")
        problem = True
    if x.shape != y.shape:
        print(f"{colors.RED}    **PROBLEM**: x.shape ({x.shape}) != y.shape ({y.shape}){colors.RESET}")
        problem = True

    nx = x.shape[0]

    #Compute the time delay (argmax of cross-correlation) in samples
    if DEBUG: print("    calling estimate_time_shift...")
    short_len = nx//10       # 20-minute long audio files take a while, so use a subset
    dt = estimate_time_shift(x[0:short_len], y[0:short_len])
    if dt != 0:
        print(f"{colors.RED}    **PROBLEM**: Estimated time shift of {dt} samples from input to target.{colors.RESET}")
        problem = True

    if not problem:
        print(f" {colors.GREEN}  Looks good! :-) {colors.RESET}")

#### WORK IN PROGRESS ####
sys.exit(1)

### The following was just copied from scipy.signal to show how to compute a
#   cross-correlation
sig = np.repeat([0., 1., 1., 0., 1., 0., 0., 1.], 128)
sig_noise = sig + np.random.randn(len(sig))
corr = signal.correlate(sig_noise, np.ones(128), mode='same') / 128

clock = np.arange(64, len(sig), 128)
fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, sharex=True)
ax_orig.plot(sig)
ax_orig.plot(clock, sig[clock], 'ro')
ax_orig.set_title('Original signal')
ax_noise.plot(sig_noise)
ax_noise.set_title('Signal with noise')
ax_corr.plot(corr)
ax_corr.plot(clock, corr[clock], 'ro')
ax_corr.axhline(0.5, ls=':')
ax_corr.set_title('Cross-correlated with rectangular pulse')
ax_orig.margins(0, 0.1)
fig.tight_layout()
plt.show()

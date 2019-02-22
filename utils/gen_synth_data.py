#! /usr/bin/env python3
"""
gen_synth_data.py
Author: Scott H. Hawley

Description:
Rather than generating on the fly, this will pre-generate a large number of
(random) input-output pairs, with the knob settings notated in the filename of the
target output audio file.

This generator defaults to operating in parallel using all available processors.
For serial execution, set parallel=False in the code below.

Current default creates about 30 GB of audio in a few minutes when 12 processors are available
"""
import numpy as np
import random
import os, sys
import time
import multiprocessing as mp
from functools import partial
import argparse
# import the signaltrain routines needed
sys.path.append(os.path.abspath('../'))
from helpers import audio, io_methods


def gen_one_io_pair(t, x, sr, effect, settings_per, log_interval, num_files, file_i):
    """
    Produces input & output (target) audio clips, at one (random) knob setting for the whole thing
    This routine is called many times, either in serial or parallel.

    Inputs:
        t: range of time values used in function synthesis, for each clip
        x: pre-allocated storage for full input signal
        sr: sample rate in Hz
        effect: a member of the audio.Effect class
        settings_per:  May be None.  # of settings per knob. if None, use random value
        log_interval: how often to print status messages
        num_clips: number of clips
        file_i: an index number denoting which (random) audio clip this will be
    Outputs:
        Two .wav files, one for the input and one for the target output from the effect
        The filename of the target audio will include the knob settings, in the order they appear in effect.
           Thus the filename will need to be parsed to obtain these values for network training & inference.
    """
    # create input audio
    clip_length = t.shape[0]
    num_clips = x.shape[0] // clip_length
    for clip_i in range(num_clips):
        ibgn, iend = clip_i * clip_length, (clip_i+1)*clip_length
        chooser = np.random.choice([0,1,2,4,6,7])
        # added conditional normalization, to avoid any possible rescaling errors during training later
        tmp = audio.synth_input_sample(t, chooser)
        x[ibgn:iend] = tmp
        tmpmax = max( np.max(tmp), abs(np.min(tmp)))
        if tmpmax > 1.0:
            x[ibgn:iend] /= tmpmax


    # Decide where this data is going
    path = "Train/" if file_i/num_files < 0.8 else "Val/"

    # generate knob setting(s) -- one setting for the whole signal.
    if (path == "Val/") or (settings_per is None):   # randomly choose knob settings
        #tmp = audio.random_ends(len(effect.knob_ranges))-0.5  # normalized knob values, -.5 to .5
        knobs_nn = np.random.rand(len(effect.knob_ranges))-0.5  # uniform distribution of knob values
        knobs_wc = effect.knobs_wc(knobs_nn)           # 'physical' knob values in "world coordinates" of the effect
    else:                                           # sequentially choose knob settings
        knobs_wc = audio.int2knobs(file_i, effect.knob_ranges, settings_per)
    #print("file ",file_i,", knobs_wc = ",knobs_wc)

    # We need to enforce a certain number of significant digits to ensure reproducability (after we read the files back in)
    # The easiest way to do this is to print to string, and then convert back to values
    #  Plus we'll save the file notating the values in the effect's own unit 'coordinate system,' since that's likely how users will record data in the future
    knobs_sigfigs, knobs_str = [], ''
    for k in range(len(knobs_wc)):
        k_str = '%s' % float('%.4g' % knobs_wc[k])
        knobs_sigfigs.append(float(k_str))   # save the values to pass to the effect
        knobs_str += "__"+k_str              # save the strings to use in the filename

    # Call the effect (on the entire audio stream at once, not chunk by chunk)
    y, x = effect.go_wc(x, knobs_sigfigs)
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # save files
    filename_in = path + "input_"+str(file_i)+ "_.wav"  # note the extra _ before the .wav. That ensures the input filenames sort in the same order as the targets
    filename_target = path + "target_"+str(file_i)+"_"+effect.name + knobs_str + ".wav"

    if (file_i % log_interval == 0):   # status message
        print("file_i = ",file_i,"/",num_files,", path = ",path,", filename_in =",filename_in, "target =",filename_target)

    audio.write_audio_file(filename_in, x, sr)
    audio.write_audio_file(filename_target, y, sr)

    return


def gen_synth_data(args):

    # Parse command line arguments
    num_files = args.num
    sr = args.sr
    settings_per = args.sp
    signal_length = int(args.dur * sr)
    file_indices = range(num_files)
    if 'comp_4c' == args.effect:
        effect = audio.Compressor_4c()
    elif 'comp' == args.effect:
        effect = audio.Compressor()
    else:
        print("Sorry, not set up to work for other effects")
        sys.exit(1)

    train_val_split = 0.8  # between 0 and 1, below number will be train, rest will be val 0.8 means 80-20 split
    if settings_per is not None:  # evenly cover knob values in Train
        num_train_files = int( settings_per**len(effect.knob_ranges) ) # Evenly spaces settings
        num_files = int(num_train_files / train_val_split)
        print("Evenly spacing",settings_per,"settings across",len(effect.knob_ranges),end="")
        print(", for",num_train_files,"files in Train and",num_files,"total files")

    # Make sure Test & Val directories exist
    for dir in ["Train","Val"]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Compute a few auxiliary variables
    # We will end up concatenating a number of "clips" which have some length
    # We'll adjust the length of total audio based on number of clips
    clip_length = 4096
    num_clips = int(np.ceil(signal_length/clip_length))
    print("Number of ",clip_length,"-length clips per file: ",num_clips,sep="")
    signal_length = clip_length * num_clips

    # Set up some array storage we'll use multiple times
    t = np.arange(clip_length,dtype=np.float32) / sr # time indeices
    x = np.zeros(signal_length,dtype=np.float32)

    # Loop over the number of audio files to generate
    log_every = 100
    wrapper = partial(gen_one_io_pair, t, x, sr, effect, settings_per, log_every, num_files)
    parallel = True
    if parallel:
        # spawn across many processes
        num_procs = mp.cpu_count()
        print("Splitting",num_files,"jobs across",num_procs,"processes")
        pool = mp.Pool(num_procs)
        indices = range(num_files)
        results = pool.map(wrapper, indices)  # Farm out list of files#'s to different procs
        pool.close()
        pool.join()
    else:
        for file_i in range(num_files):
            wrapper(file_i)
    return


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    parser = argparse.ArgumentParser(description="Generate synthetic data. Train will have knob-values equally spaced, Val will be random")
    parser.add_argument('-d', '--dur', type=float, help='Duration of audio files, in seconds (approximate)', default=5)
    parser.add_argument('--sp', type=int, help='Settings per knob (in Train set)', default=None)
    parser.add_argument('-n', '--num', type=int, help='Number of audio files to generate (turned off if --sp option enabled)', default=20000)
    parser.add_argument('-e', '--effect', help='Name of effect to use', default="comp_4c")
    parser.add_argument('--sr', type=int, help='Sampling rate', default=44100)
    args = parser.parse_args()
    if args.sp is None:
        print("Warning: Defaults will generate approximately",33.7*args.num/20000*args.dur/5,"GB of audio in Train/ and Val/ directories")
    gen_synth_data(args)

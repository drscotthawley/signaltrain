#! /usr/bin/env python3
"""
gen_dataset.py
Author: Scott H. Hawley

Description:
Rather than generating on the fly during training, this will pre-generate a large number of
input-output pairs, with the knob settings notated in the filename of the
target output audio file.

Is also able to read input audio (e.g. music) from a directory specified by --inpath.
This path should *already* be split into Train and Val (and possibly Test) subdirectories
# NOTE: currently this program EITHER uses pre-fab inputs OR generates new ones.
# Thus to have a dataset with *both* music and random test tones, run this program twice,
# with and without the --inpath option .

This generator defaults to operating in parallel using all available processors.
For serial execution (e.g. for debugging), set parallel=False in the code below.

Current defaults create about 30 GB of audio in a few minutes when 12 processors are available
"""
import numpy as np
import random
import os
import sys
import glob
import time
import multiprocessing as mp
from functools import partial
import argparse
# import the signaltrain routines needed
import signaltrain as st
import random


parallel = True      # parallel execution. May need to set to False for debugging
dtype = np.float32   # Note: both Numba (for effects) and scipy.wavfile.write need float32, not float16

def gen_one_io_pair(name, t, x, sr, effect, settings_per, log_interval, infile_list, num_outfiles, start_output_i, outfile_i):
    """
    One instance to be called in trivially-parallel implentation

    Produces input & output (target) audio clips, at one (random) knob setting for the whole thing
    This routine is called many times, either in serial or parallel.


    Inputs:
        name: name of the dataset (output subdirectory )
        t: range of time values used in function synthesis, for each clip
        x: pre-allocated storage for full input signal
        sr: sample rate in Hz
        effect: a member of the audio.Effect class
        settings_per:  May be None.  # of settings per knob. if None, use random value
        log_interval: how often to print status messages
        num_clips: number of clips
        outfile_i: an index number denoting which (random) audio clip this will be
    Outputs:
        Two .wav files, one for the input and one for the target output from the effect
        The filename of the target audio will include the knob settings, in the order they appear in effect.
           Thus the filename will need to be parsed to obtain these values for network training & inference.
    """
    outpath = name+'/'

    # Decide where this data is coming from
    if infile_list is not None:             # use pre-existing input files
        # read audio from file on the list
        infile_i =  outfile_i % len(infile_list)  # sequentially walk through and 'wrap-around' end of infile list
        infilename = infile_list[infile_i]

        clip_len = len(x)                         # signal length is stored in x from earlier

        x, sr = st.audio.read_audio_file(infilename, sr=sr, dtype=dtype) # overwrite x by reading audio

        # but only use a random subset of x, given by len(t) (which was set by --dur)

        # grab a random part of the file
        if clip_len >= len(x):  # unless there's not enough audio in the file to justify this
            randi = 0
            clip_len = len(x)
        else:
            randi = random.randint(0, x.shape[0]-clip_len-1) # random index at which to start the clip
        x = x[randi:randi+clip_len]

        # destination output dir: base it on what's the input path
        if 'Train' in infilename:
            outpath += 'Train/'
        elif 'Val' in infilename:
            outpath += 'Val/'
        elif 'Test' in infilename:
            outpath += 'Test/'
            if not os.path.exists(outpath):
                os.makedirs(outpath)

    else:                                 # synthesize new input
        # Input audio: synthesize or read from file
        clip_length = t.shape[0]
        num_clips = x.shape[0] // clip_length
        for clip_i in range(num_clips):
            ibgn, iend = clip_i * clip_length, (clip_i+1)*clip_length
            chooser = np.random.choice([0,1,2,4,6,7,8,9])   # skipping 5="bunch of spikes"

            # added conditional normalization, to avoid any possible rescaling errors during training later
            tmp = st.audio.synth_input_sample(t, chooser)
            x[ibgn:iend] = tmp
            tmpmax = max( np.max(tmp), abs(np.min(tmp)))
            if tmpmax > 1.0:
                x[ibgn:iend] /= tmpmax

        # and decide where to send it (for synthesized audio)
        if outfile_i/num_outfiles > 0.8:
            outpath += 'Val/'
        else:
            outpath += 'Train/'

    # generate knob setting(s) -- one setting for the whole signal "streamed target"
    nk = len(effect.knob_ranges)   # numknobs
    if (('Train' not in outpath) and ('Val' not in outpath)) or (settings_per is None) or (outfile_i >= settings_per**nk):
        # Then randomly choose knob settings
        knobs_nn = np.random.rand(nk)-0.5  # uniform distribution of knob values
        knobs_wc = effect.knobs_wc(knobs_nn)           # 'physical' knob values in "world coordinates" of the effect
    else:                                           # sequentially choose knob settings
        knobs_wc = st.audio.int2knobs(outfile_i, effect.knob_ranges, settings_per)
        #print(f"file #{outfile_i}, settings_per = {settings_per}, knobs_wc = {knobs_wc}")

    # We need to enforce a certain number of significant digits to ensure reproducability (after we read the files back in)
    # The easiest way to do this is to print to string, and then convert back to values
    #  Plus we'll save the file notating the values in the effect's own unit 'coordinate system,' since that's likely how users will record data in the future
    knobs_sigfigs, knobs_str = [], ''
    for k in range(len(knobs_wc)):
        k_str = '%s' % float('%.4g' % knobs_wc[k])
        knobs_sigfigs.append(float(k_str))   # save the values to pass to the effect
        knobs_str += "__"+k_str              # save the strings to use in the filename

    # Actually run the effect (on the entire audio stream at once, not chunk by chunk)
    y, x = effect.go_wc(x, knobs_sigfigs)

    # save files
    out_idx = start_output_i + outfile_i  # hey, don't overwrite existing files
    outfilename_input = outpath + "input_"+str(out_idx)+ "_.wav"  # note the extra _ before the .wav. That ensures the input filenames sort in the same order as the targets
    outfilename_target = outpath + "target_"+str(out_idx)+"_"+effect.name + knobs_str + ".wav"

    if (outfile_i % log_interval == 0):   # status message
        if infile_list is not None:
            print("orig input file = ",infilename)
        print("outfile_i = ",outfile_i,"/",num_outfiles,", outpath = ",outpath,", outfilename_input = ",outfilename_input, ", target = ",outfilename_target,sep="")

    st.audio.write_audio_file(outfilename_input, x.astype(dtype, copy=False), sr)
    st.audio.write_audio_file(outfilename_target, y.astype(dtype, copy=False), sr)

    return


def gen_synth_data(args):

    # Parse command line arguments
    name = args.name
    num_outfiles = args.num
    sr = args.sr
    settings_per = args.sp
    signal_length = int(args.dur * sr)
    outfile_indices = range(num_outfiles)
    inpath = args.inpath

    if 'comp_4c' == args.effect:
        effect = st.audio.Compressor_4c()
    elif 'comp' == args.effect:
        effect = st.audio.Compressor()
    elif 'comp_t' == args.effect:
        effect = st.audio.Comp_Just_Thresh()
    elif 'comp_large' == args.effect:
        effect = st.audio.Compressor_4c_Large()
    else:
        print("Sorry, not set up to work for other effects")
        sys.exit(1)

    train_val_split = 0.8  # between 0 and 1, below number will be train, rest will be val 0.8 means 80-20 split
    if settings_per is not None:  # evenly cover knob values in Train
        num_train_files = int( settings_per**len(effect.knob_ranges) ) # Evenly spaces settings
        if (inpath is None) or (('Train' not in inpath) and ('Val' not in inpath)):
            num_outfiles = int(num_train_files / train_val_split)
        else:
            num_outfiles = num_train_files
        print("Evenly spacing",settings_per,"settings across",len(effect.knob_ranges)," knob(s)",end="")
        print(", for",num_train_files,"files in Train and",num_outfiles,"total files")

    # Make sure name, Test & Val directories exist
    for dir in [name, name+"/Train",name+"/Val"]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # create an effect_info.ini file in the new dataset directory
    with open(name+"/effect_info.ini", "w") as info_file:
        print("[effect]", file=info_file)
        print(f"name = {effect.name}",file=info_file)
        print(f"knob_names = {effect.knob_names}",file=info_file)
        print(f"knob_ranges = {effect.knob_ranges.tolist()}",file=info_file)

    # for synthed inputs only:
    # Compute a few auxiliary variables
    # We will end up concatenating a number of "clips" which have some length
    # We'll adjust the length of total audio based on number of clips
    clip_length = 4096
    num_clips = int(np.ceil(signal_length/clip_length))
    signal_length = clip_length * num_clips

    # Set up some array storage we'll use multiple times
    t = np.arange(clip_length,dtype=dtype) / sr # time indeices
    x = np.zeros(signal_length,dtype=dtype)


    # If input files are specified via --inpath
    infile_list = None
    if inpath != None:
        infile_list = glob.glob(inpath+"/*.wav")
        infile_list += glob.glob(inpath+"/*/*.wav")
        infile_list = [ x for x in infile_list if "target" not in x ]  # remove any 'target' audio
        print("\ninfile_list =",infile_list)
    else:
        print("Number of ",clip_length,"-length clips per synthesized input file: ",num_clips,sep="")


    # was having problems with existing files getting overwritten
    num_already_there = len(glob.glob(name+"/*/input*"))  # count the number of pre-existing input files
    start_output_i = num_already_there  # we're zero indexed

    # Loop over the number of audio files to generate
    log_every = 100
    wrapper = partial(gen_one_io_pair, name, t, x, sr, effect, settings_per, log_every, infile_list, num_outfiles, start_output_i)
    if parallel:
        # spawn across many processes
        num_procs = mp.cpu_count()
        print("Splitting",num_outfiles,"jobs across",num_procs,"processes")
        pool = mp.Pool(num_procs)
        indices = range(num_outfiles)
        results = pool.map(wrapper, indices)  # Farm out list of files#'s to different procs
        pool.close()
        pool.join()
    else:
        for outfile_i in range(num_outfiles):
            wrapper(outfile_i)
    return


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    parser = argparse.ArgumentParser(description="Generate synthetic data. Train will have knob-values equally spaced, Val will be random",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('name', help='Name of the dataset (creates new subdirectory)')
    parser.add_argument('-d', '--dur', type=float, help='Duration of generated input (& ouput) files, in seconds (approximate)', default=5)
    parser.add_argument('--sp', type=int, help='Settings per knob (in Train set)', default=None)
    parser.add_argument('-n', '--num', type=int, help='Number of audio files to generate (turned off if --sp option enabled)', default=20000)
    parser.add_argument('-e', '--effect', help='Name of effect to use', default="comp_4c")
    parser.add_argument('--inpath', help='Can read audio input files from here ', default=None)
    parser.add_argument('--sr', type=int, help='Sampling rate', default=44100)

    args = parser.parse_args()
    if args.sp is None:
        print("Warning: Defaults will generate approximately",33.7*args.num/20000*args.dur/5,"GB of audio in Train/ and Val/ directories")
    gen_synth_data(args)

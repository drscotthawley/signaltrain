#! /usr/bin/env python3
import argparse
import sys
import os
import glob
import librosa
import shutil

"""
Trying to resample audio on the fly for each run takes so long, one might
as well just do the whole dataset at once

Works like this:

   $ ./resample_dataset.py <dir> <sr>

which will create a directory named <dir>+_+<sr>Hz, that preserves the structure
of the original.

Note on librosa usage: It can be slow, but when you're resampling, it's no slower
than what you'd have to do anyway.
"""

import os,sys


parser = argparse.ArgumentParser(description="Resample a whole dataset")
parser.add_argument('dir', help='Directory of dataset')
parser.add_argument('sr', type=int, help='Sampling rate')
args = parser.parse_args()

main_dir = args.dir
sr = args.sr

new_main_dir = main_dir + '_'+str(sr)+"Hz"    #  name of new (resampled) dataset

if os.path.exists(new_main_dir):
    shutil.rmtree(new_main_dir) # wipe any previous directory named new_main_dir
os.makedirs(new_main_dir)   # create newmaindir

count = 0
for (dirname, dirs, files) in os.walk(args.dir):
    new_dirname = dirname.replace(main_dir, new_main_dir)  # dirname already includes full path
    # create new dirname if it doesn't exist
    print(f"\n{dirname} -> {new_dirname}")
    if not os.path.exists(new_dirname):
        os.makedirs(new_dirname)
    for filename in files:
        print(f"         {dirname}/{filename} -> {new_dirname}/{filename}")
        in_path = dirname+'/'+filename
        out_path = new_dirname+'/'+filename
        if ('.wav' in filename) or ('.mp3' in filename) or ('.WAV' in filename):
            signal, rate = librosa.load(in_path, sr=sr, res_type='kaiser_fast') # Librosa's reader is incredibly slow. do not use
            librosa.output.write_wav(out_path, signal, sr, norm=False)
        else:
            shutil.copy(in_path, out_path)

#!/usr/bin/env python3

# VERY simple utility for doing random Train/Val split of data files in current directory

import os, sys, glob
import random
import shutil

# First, make sure Train/ and Val directories exist
for dir in ["Train","Val"]:
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:   # if they do, then pull anything currently in them out into the current/main directory
        for file in glob.glob(dir+'/*'):
            shutil.move(file,'.')


# Now, with all files in current directory, split files into Train and Val

path = "."
input_filenames = sorted(glob.glob(path+'/'+'input_*'))
target_filenames = sorted(glob.glob(path+'/'+'target_*'))

split_prob = 0.8  # probability of being sent to Train/

#  random.seed(1) # To get the same split every time, uncomment this line.

for i in range(len(input_filenames)):
    print(i)
    if random.random() < split_prob:
        dstdir = 'Train/'
    else:
        dstdir = 'Val/'

    shutil.move(input_filenames[i], dstdir)
    shutil.move(target_filenames[i], dstdir)

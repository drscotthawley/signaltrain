# -*- coding: utf-8 -*-
__author__ = 'S.H. Hawley'

"""
Routines related to audio datasets, either consisting of files or generated/synthesized
on the fly
"""


# imports
import numpy as np
import torch
import os
import glob
from torch.utils.data import Dataset
from . import audio
import sys
import random


def do_augment(x, y, rand_invert=True, mult_some=False, add_some=False):
    """
    A few simple hacks for data augmentation, to make the dataset go further.
    Nothing fancy: Don't want to mess up the target data if this is a file-based dataset.
    Note that time translations are effectively already handled by the data generators (below).
    """
    if rand_invert and np.random.choice([True,False]):
        # randomly flip phases of both input & target
        x, y = -x, -y

    # Below are ugmentations that only affect the lookback part of the input x
    lookback = x.size - y.size

    if mult_some and np.random.choice([True,False]):
        # like "cutout" or "salt & pepper". here we randomly multiply isolated samples by random factors.
        # NOTE: this will end up adding lots of high-freq info in the STFT and may not be adviseable.
        fraction = 0.2 # mess with this fraction of number of samples in the lookback window
        n = int(lookback * fraction)                     # number of samples to modify
        indices = np.random.randint(0, high=lookback, size=n) # indices where mod will occur
        mults = (2*np.random.rand(n)-1).astype(x.dtype, copy=False)  # random multiplicative factor [-1..1]
        x[indices] = x[indices] * mults

    if add_some and np.random.choice([True,False]):
        # similar to mult_some, but we add verrry small noise to some points
        fraction = 0.3
        n = int(lookback * fraction)                     # number of samples to modify
        indices = np.random.randint(0, high=lookback, size=n)
        tiny_fac = np.max(x) / 1e6
        adds = ( tiny_fac*(2*np.random.rand(n)-1) ).astype(x.dtype, copy=False)
        x[indices] = x[indices] + adds
    return x, y


def worker_init(worker_id):
    """
    used with PyTorch DataLoader so that we can grab random bits of files or
    synth random input data on the fly
    Without this you get the same thing every epoch
    """
    # NOTE that this current implementation prevents strict reproducability
    np.random.seed()


class AudioFileDataSet(Dataset):
    """
    Read from premade audio files.  Is a subclass of PyTorch Dataset so you can use DataLoader
    TODO: self, how about writing more documentation for this?
    """
    def __init__(self, chunk_size, effect, sr=44100, path="./Train/", datapoints=8000, \
        dtype=np.float32, preload=True, rerun=False, y_size=None, augment=True,
        align_end=True, view_of=None, compand=False):
        """
        view_of: as a memory-saving experiment, pass another dataset.
        """
        super(AudioFileDataSet, self).__init__()

        self.chunk_size = chunk_size
        self.effect = effect  # The only reason we still need the effect defined (even though we're reading files) is to get the RANGES for the knobs
        self.sr = sr
        self.path  = path
        self.dtype = dtype
        self.datapoints = datapoints
        self.preload = preload
        self.align_end = align_end # found that some of our audio files have starting points that are not aligned, but their einding points are.
        self.rerun_effect = rerun  # a hack to avoid causality issues at chunk boundaries
        self.compand = compand
        if y_size is None:
            self.y_size = chunk_size
        else:
            self.y_size = y_size
        self.augment = augment   # data augmentation: randomly invert phase
        self.processed_dir = ''
        self.num_knobs = 0
        '''
        # Loading raw audio files (".wav") is incredibly slow. Much fast to preprocess and save in another format
        check_preproc = False
        if check_preproc:
            self.processed_dir = 'processed_audio/'
            self.process_audio()
        '''

        if view_of is None:
            print("view_of = None. Original data")
            # get a list of available files.  Note that knob settings are included to the target filenames
            self.input_filenames = sorted(glob.glob(self.processed_dir+self.path+'/'+'input_*'))
            self.target_filenames = sorted(glob.glob(self.processed_dir+self.path+'/'+'target_*'))
            print("AudioFileDataSet: Found",len(self.input_filenames),"input files and",
                len(self.target_filenames)," target files in path",self.path)
            assert len(self.input_filenames) == len(self.target_filenames)   # TODO: One can imagine a scheme with multiple targets per input

            print("  AudioFileDataSet: Check to make sure input & target filenames sorted together in the same order:")
            for i in range(min(10, len(self.input_filenames))):
                print("      i =",i,", input_filename =",os.path.basename(self.input_filenames[i]),\
                  ", target_filename =",os.path.basename(self.target_filenames[i]))

            if self.preload:  # load data files into memory first
                self.preload_audio()
        else:
            print(" *** Warning, using only a view of earlier dataset")
            self.x, self.y, self.knobs = view_of.x, view_of.y, view_of.knobs
            self.num_knobs = view_of.num_knobs


    def preload_audio(self):
        # This is much faster than reading files anew each epoch, but comes at the cost of assuming a uniform dataset.
        # Alternative is to have workers load files at each epoch: that saves memory but is WAY slower that preloading
        print("    Preloading audio files for this dataset...")
        files_to_load = min(100000, len(self.input_filenames))   # min / trap to avoid memory errors
        audio_in, audio_targ, knobs_wc = self.read_one_new_file_pair()  # read one file for sizing
        dur = len(audio_in)/self.sr
        self.num_knobs = len(knobs_wc)
        self.x, self.y = [],[]
        #self.x = np.zeros((files_to_load,len(audio_in) ),dtype=self.dtype)
        #self.y = np.zeros((files_to_load,len(audio_targ) ),dtype=self.dtype)
        self.knobs = np.zeros((files_to_load, self.num_knobs ),dtype=self.dtype)
        print_every = files_to_load//10 if 0!= files_to_load//10 else 1
        for i in range(files_to_load):
            tmp_x, tmp_y,  self.knobs[i] = self.read_one_new_file_pair(idx=i)

            if ((i+1) % print_every == 0) or (i+1 == files_to_load):
                print("\r       i = ",i+1,"/",files_to_load," len x =",len(tmp_x), "dur=",len(tmp_x)/44100.0/60," min", " len y =",len(tmp_y))

            if (len(tmp_x) != len(tmp_y)):
                print("  ***Warning: Length mismatch. input & output filenames:",
                self.input_filenames[i],self.target_filenames[i])
                if self.align_end:
                    minlen = min(len(tmp_x), len(tmp_y))
                    print("   Aligning to ends:")
                    print("      Before alignment: tmp_x.shape, tmp_y.shape = ",tmp_x.shape, tmp_y.shape)
                    tmp_x, tmp_y = tmp_x[-minlen:], tmp_y[-minlen:]
                    print("      After alignment:  tmp_x.shape, tmp_y.shape = ",tmp_x.shape, tmp_y.shape)

            if self.effect.is_inverse:
                tmp_x, tmp_y = tmp_y, tmp_x         # for effects that reverse 'input' and 'output' (for de-____ effects)

            # add audio to the lists that make up the datasets
            self.x.append(tmp_x)
            self.y.append(tmp_y)

        print("    ...finished preloading")

    def __len__(self):
        return self.datapoints

    def process_audio(self):  # TODO: not used yet following torchaudio
        """ Render raw audio as pytorch-friendly file. TODO: not done yet.
        """
        if os.path.exists(self.processed_dir):
            return

        # get a list of available audio files.  Note that knob settings are included to the target filenames
        input_filenames = sorted(glob.glob(self.path+'/'+'input_*'))
        self.target_filenames = sorted(glob.glob(self.path+'/'+'target_*'))
        assert len(input_filenames) == len(target_filenames)   # TODO: One can image a scheme with multiple targets per input
        print("Dataset: Found",self.__len__(),"raw audio i-o pairs in path",self.path)


    def parse_knob_string(self, knob_str, ext=".wav"):  # given target filename, get knob settings
        """ By convention, we will use double-underscores in the filename before each knob setting,
            and these should be the last part of the filename before the extension
            Nowhere else in the filename should double underscores appear.
            Example: 'target_9400_Compressor_4c__-10.95__3.428__0.005043__0.01308.wav'
        """
        knob_list = knob_str.replace(ext,'').split('__')[1:] # strip ext, and throw out everything before first __'s
        knobs = np.array([float(x) for x in knob_list], dtype=self.dtype)  # turn list of number-strings into float numpy array
        return knobs


    def read_one_new_file_pair(self, idx=None):
        """
        read from input-target audio files, and parse target audio filename to get knob setting
        Inputs:
            idx (optional): index within the list of filenames to read from
        """
        if idx is None:
            idx = np.random.randint(0,high=len(self.input_filenames)) # pick a file at random

        audio_in, sr = audio.read_audio_file(self.input_filenames[idx], sr=self.sr, fix_and_overwrite=True)
        audio_targ, sr = audio.read_audio_file(self.target_filenames[idx], sr=self.sr, fix_and_overwrite=True)

        # parse knobs from target filename
        knobs_wc = self.parse_knob_string(self.target_filenames[idx])

        '''# Keeping this in code in case I want to switch it on for diagnostic reasons later
         # only rerun effect on chunks. see get_single_chunk() below
        if self.rerun_effect: # run effect on entire file; this for checking/diagnostic only; can ignore for typical uses
            audio_orig = np.copy(audio_targ)
            audio_targ, audio_in = self.effect.go_wc(audio_in, knobs_wc)

            audio_diff = audio_targ - audio_orig
            if np.max(np.abs(audio_diff)) > 1e-6:  # output log files for when this makes a difference
                audio.write_audio_file('audio_in_'+str(idx)+'.wav', audio_in, sr=44100)
                audio.write_audio_file('audio_orig_'+str(idx)+'.wav', audio_orig, sr=44100)
                audio.write_audio_file('audio_targ_'+str(idx)+'.wav', audio_targ, sr=44100)
                audio.write_audio_file('audio_diff_'+str(idx)+'.wav', audio_diff, sr=44100)
            '''

        if self.compand:
            audio_in = audio.mu_compand(audio_in)
            audio_targ = audio.mu_compand(audio_targ)

        return audio_in, audio_targ, knobs_wc


    def get_single_chunk(self):
        """
        Grabs audio and knobs either from files or from preloaded buffer(s)
        """
        if self.preload:  # This will typically be the case
            i = np.random.randint(0,high=len(self.x))  # pick a random line from preloaded audio
            in_audio, targ_audio, knobs_wc = self.x[i], self.y[i], self.knobs[i]  # note these might be, e.g. 10 seconds long
        else:
            in_audio, targ_audio, knobs_wc = self.read_one_new_file_pair() # read x, y, knobs

        # Grab a random chunk from within total audio nfile
        assert len(in_audio) > self.chunk_size, f"Error: len(in_audio)={len(in_audio)}, must be > self.chunk_size={self.chunk_size}"
        ibgn = np.random.randint(0, len(in_audio) - self.chunk_size)
        x_item = in_audio[ibgn:ibgn+self.chunk_size]
        y_item = targ_audio[ibgn:ibgn+self.chunk_size]

        if self.rerun_effect:  # re-run the effect on this chunk , and replace target audio
            y_item, x_item = self.effect.go_wc(x_item, knobs_wc)   # Apply the audio effect

        y_item = y_item[-self.y_size:]   # Format for expected output size

        # normalize knobs for nn usage
        kr = self.effect.knob_ranges   # kr is abbribation for 'knob ranges'
        knobs_nn = (knobs_wc - kr[:,0])/(kr[:,1]-kr[:,0]) - 0.5

        if self.augment:
            x_item, y_item = do_augment(x_item, y_item)

        return x_item.astype(self.dtype, copy=False), y_item.astype(self.dtype, copy=False), knobs_nn.astype(self.dtype, copy=False)

    # required part of torch.Dataset class.  This is how DataLoader gets a new piece of data
    def __getitem__(self, idx):  # we ignore idx and grab a random bit from a random file
        #if self.recycle:
        #    return self.x[idx], self.y[idx], self.knobs[idx]
        return self.get_single_chunk()



class SynthAudioDataSet(Dataset):
    """
    Generates synthetic audio data on the fly

    Note that in  PyTorch DataLoader, the random number generator is reset (seeded to a fixed value)
    for each worker at the beginning of each EPOCH.
    Which means that 'random' data items will be REPEATED from epoch to epoch when using the PyTorch DataLoader
    In this sense, it can therefore simulate having a finite dataset.

    To prevent this, one can pass the DataLoader a worker_init_fn that sets the random seed.
       See https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    def __init__(self, chunk_size,  effect, sr=44100, datapoints=8000, dtype=np.float32,
        recycle=False, y_size=None, augment=True):
        super(SynthAudioDataSet, self).__init__()
        self.chunk_size = chunk_size
        self.effect = effect
        self.sr = sr
        self.datapoints = datapoints
        self.dtype = dtype
        self.recycle = recycle
        self.num_knobs = len(effect.knob_names)
        self.y_size = chunk_size if (y_size is None) else y_size
        self.augment = augment   # data augmentation: randomly invert phase

        # preallocate an array of time values across one chunk for use with audio synth functions
        self.t = np.arange(chunk_size, dtype=np.float32) / sr

        print("SynthAudioDataSet: synthetic/generated data")

        if recycle:  # keep the same data over & over (Useful for monitoring Validation set)
            print("Setting up recycling (static data) for this dataset. This may take a short while...")
            self.x = np.zeros((datapoints,chunk_size), dtype=self.dtype)
            self.y = np.zeros((datapoints,self.y_size ), dtype=self.dtype)
            self.knobs = np.zeros((datapoints, self.num_knobs), dtype=self.dtype)
            for i in range(datapoints):
                self.x[i], self.y[i], self.knobs[i] = self.gen_single_chunk()
            print("...done")

    def __len__(self):
        return self.datapoints

    def __getitem__(self,idx):  # Basic PyTorch operation with DataLoader
        if self.recycle:
            return self.x[idx], self.y[idx], self.knobs[idx]

        x, y, knobs = self.gen_single_chunk()
        return x.astype(self.dtype, copy=False)[-self.chunk_size:], y[-self.y_size:], knobs.astype(self.dtype, copy=False)

    def gen_single_chunk(self, chooser=None, knobs=None):
        """
        create a single time-series of input and target output, all with the same knobs setting
        """
        if chooser is None:
            chooser = np.random.choice([0,1,2,4,6,7])  # for compressor
            #chooser = np.random.choice([0,1,3,4,6,10,2,7,9]) # for compressor & more
            #chooser = 4  # for just step response ('cheating')
            #chooser = np.random.choice([1,3,5,6,7])  # for echo

        x = audio.synth_input_sample(self.t, chooser)

        if knobs is None:
            knobs = audio.random_ends(len(self.effect.knob_ranges))-0.5  # inputs to NN, zero-mean...except we emphasize the ends slightly

        y, x = self.effect.go(x, knobs)   # Apply the audio effect

        y = y[-self.y_size:]    # shrink output size

        if self.augment:
            x, y = do_augment(x, y)

        return x, y, knobs


# EOF

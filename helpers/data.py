# -*- coding: utf-8 -*-
__author__ = 'S.H. Hawley'

# imports
import numpy as np
import torch
import os
import glob
from torch.utils.data import Dataset
from helpers import audio


def worker_init(worker_id): # used with PyTorch DataLoader
    np.random.seed()  # TODO: note that this current implementation prevents reproducability


class AudioFileDataSet(Dataset):
    """
    Read from premade audio files
    """
    def __init__(self, chunk_size, effect, sr=44100, path="./Train/", datapoints=8000, \
        dtype=np.float32, preload=True, skip_factor=0.75, rerun=False, y_size=None):
        super(AudioFileDataSet, self).__init__()

        self.chunk_size = chunk_size
        self.effect = effect  # The only reason we still need the effect defined (even though we're reading files) is to get the RANGES for the knobs
        self.sr = sr
        self.path  = path
        self.dtype = dtype
        self.datapoints = datapoints
        self.preload = preload
        self.skip_over = int(self.chunk_size * skip_factor)  # This will overwrite the first part of y with x
        self.rerun_effect = rerun  # a hack to avoid causality issues at chunk boundaries
        if y_size is None:
            self.y_size = chunk_size
        else:
            self.y_size = y_size
        self.processed_dir = ''
        '''
        # Loading raw audio files (".wav") is incredibly slow. Much fast to preprocess and save in another format
        check_preproc = False
        if check_preproc:
            self.processed_dir = 'processed_audio/'
            self.process_audio()
        '''

        # get a list of available files.  Note that knob settings are included to the target filenames
        self.input_filenames = sorted(glob.glob(self.processed_dir+self.path+'/'+'input_*'))
        self.target_filenames = sorted(glob.glob(self.processed_dir+self.path+'/'+'target_*'))
        print("AudioFileDataSet: Found",len(self.input_filenames),"i-o pairs in path",self.path)
        assert len(self.input_filenames) == len(self.target_filenames)   # TODO: One can image a scheme with multiple targets per input

        print("  AudioFileDataSet: Check to make sure input & target filenames sorted together in the same order:")
        for i in range(10):
            print("      i =",i,", input_filename =",self.input_filenames[i],", target_filename =",self.target_filenames[i])

        if self.preload:  # load data files into memory first
            self.preload_audio()

    def preload_audio(self):
        print("    Preloading audio files for this dataset...")
        files_to_load = min(20000, self.datapoints//5, len(self.input_filenames))   # min to avoid memory errors
        audio_in, audio_targ, knobs_wc = self.read_one_new_file_pair()  # read one file for sizing
        self.x = np.zeros((files_to_load,len(audio_in) ),dtype=self.dtype)
        self.y = np.zeros((files_to_load,len(audio_targ) ),dtype=self.dtype)
        self.knobs = np.zeros((files_to_load, len(knobs_wc) ),dtype=self.dtype)
        for i in range(files_to_load):
            if ((i+1) % 1000 == 0) or (i+1 == files_to_load):
                print("\r       i = ",i+1,"/",files_to_load,sep="",end="")
            self.x[i], self.y[i], self.knobs[i] = self.read_one_new_file_pair(idx=i)
        if (self.skip_over > 0):
            print("\nAudioFileDataSet: We'll be skipping the first",self.skip_over,"samples in each chunk")
            self.y[:,0:self.skip_over] = self.x[:,0:self.skip_over] # overwrite first part of targets, to facilitate training
            print("Checking overwrite: diff = ",np.sum(np.abs(self.y[:,0:self.skip_over] - self.x[:,0:self.skip_over])) )
        print("    ...done preloading")

    def __len__(self):
        return self.datapoints
        #return len(self.input_filenames)

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

        audio_in, sr = audio.read_audio_file(self.input_filenames[idx], sr=self.sr)
        audio_targ, sr = audio.read_audio_file(self.target_filenames[idx], sr=self.sr)

        # parse knobs from target filename
        knobs_wc = self.parse_knob_string(self.target_filenames[idx])

        if self.rerun_effect: # run effect on entire file
            audio_orig = np.copy(audio_targ)
            audio_targ, audio_in = self.effect.go_wc(audio_in, knobs_wc)
            audio_diff = audio_targ - audio_orig
            if np.max(np.abs(audio_diff)) > 1e-6:  # output log files for when this makes a difference
                audio.write_audio_file('audio_in_'+str(idx)+'.wav', audio_in, sr=44100)
                audio.write_audio_file('audio_orig_'+str(idx)+'.wav', audio_orig, sr=44100)
                audio.write_audio_file('audio_targ_'+str(idx)+'.wav', audio_targ, sr=44100)
                audio.write_audio_file('audio_diff_'+str(idx)+'.wav', audio_diff, sr=44100)

        return audio_in, audio_targ, knobs_wc


    def get_single_chunk(self):
        """
        Grabs audio and knobs either from files or from preloaded buffer(s)
        """
        if not self.preload:
            in_audio, targ_audio, knobs_wc = self.read_one_new_file_pair() # read x, y, knobs
        else:
            i = np.random.randint(0,high=self.x.shape[0])  # pick a random line from preloaded audio
            in_audio, targ_audio, knobs_wc = self.x[i], self.y[i], self.knobs[i]  # note these might be, e.g. 10 seconds long

        # Grab a random chunk from audio
        ibgn = np.random.randint(0, len(in_audio) - self.chunk_size)
        x_item = in_audio[ibgn:ibgn+self.chunk_size]
        y_item = targ_audio[ibgn:ibgn+self.chunk_size]

        # TODO: stupid hacky hack:
        if self.rerun_effect:  # re-run the effect on this chunk , and replace target audio
            y_item, x_item = self.effect.go_wc(x_item, knobs_wc)   # Apply the audio effect

        if (self.skip_over > 0):
            y_item[0:self.skip_over] = x_item[0:self.skip_over]

        y_item = y_item[-self.y_size:]   # Format for expected output size

        # normalize knobs for nn usage
        kr = self.effect.knob_ranges
        knobs_nn = (knobs_wc - kr[:,0])/(kr[:,1]-kr[:,0]) - 0.5

        return x_item.astype(self.dtype), y_item.astype(self.dtype), knobs_nn.astype(self.dtype)


    def __getitem__(self, idx):  # we ignore idx and grab a random bit from a random file
        #if self.recycle:
        #    return self.x[idx], self.y[idx], self.knobs[idx]
        return self.get_single_chunk()


class AudioDataGenerator(Dataset):
    """
    Generates synthetic audio data on the fly

    Note that in  PyTorch DataLoader, the random number generator is reset (seeded to a fixed value)
    for each worker at the beginning of each EPOCH.
    Which means that 'random' data items will be REPEATED from epoch to epoch when using the PyTorch DataLoader
    In this sense, it can therefore simulate having a finite dataset.

    To prevent this, one can pass the DataLoader a worker_init_fn that sets the random seed.
       See https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    def __init__(self, chunk_size,  effect, sr=44100, datapoints=8000, dtype=np.float32, recycle=False):
        super(AudioDataGenerator, self).__init__()
        self.chunk_size = chunk_size
        self.effect = effect
        self.sr = sr
        self.datapoints = datapoints
        self.dtype = dtype
        self.recycle = recycle

        # preallocate an array of time values across one chunk for use with audio synth functions
        self.t = np.arange(chunk_size, dtype=np.float32) / sr

        if recycle:  # keep the same data over & over (Useful for monitoring Validation set)
            print("Setting up recycling for this dataset...")
            self.x = np.zeros((datapoints,chunk_size ),dtype=self.dtype)
            self.y = np.zeros((datapoints,chunk_size ),dtype=self.dtype)
            self.knobs = np.zeros((datapoints, len(effect.knob_names) ),dtype=self.dtype)
            for i in range(datapoints):
                self.x[i], self.y[i], self.knobs[i] = self.gen_single_chunk()
            print("...done")

    def __len__(self):
        return self.datapoints

    def __getitem__(self,idx):  # Basic PyTorch operation with DataLoader
        if self.recycle:
            return self.x[idx], self.y[idx], self.knobs[idx]

        x, y, knobs = self.gen_single_chunk()
        return x.astype(self.dtype)[-self.chunk_size:], y[-self.chunk_size:], knobs.astype(self.dtype)

    def gen_single_chunk(self, chooser=None, knobs=None):
        """
        create a single time-series of input and target output, all with the same knobs setting
        """
        if chooser is None:
            chooser = np.random.choice([0,1,2,4,6,7])  # for compressor
            #chooser = np.random.choice([1,3,5,6,7])  # for echo

        x = audio.synth_input_sample(self.t, chooser)

        if knobs is None:
            knobs = audio.random_ends(len(self.effect.knob_ranges))-0.5  # inputs to NN, zero-mean...except we emphasize the ends slightly

        y, x = self.effect.go(x, knobs)   # Apply the audio effect

        return x, y, knobs


### EOF.  Below this line is legacy code what will be removed on next commit
'''
class AudioFileDataLoader(DataLoader):
    def __init__(self, batch_size=10, requires_grad=True, device=torch.device("cuda:0"), path=None):
        super(AudioDataGenerator, self).__init__()
        self.batch_size = batch_size
        self.requires_grad = requires_grad
        self.device = device


        # preallocate memory
        #  Our input audio will appear to be smaller than our X, array. We will later generate a 'view' of the input audio to have dimensions (batch_size,chunk_size)
        # see below.  self.x = np.zeros((batch_size,chunk_size),dtype=np.float32)  # input audio broken into chunks
        self.x = np.zeros((batch_size,chunk_size),dtype=np.float32)  # TODO: this is going to get overwritten. why allocate it?
        self.y = np.zeros((batch_size,chunk_size),dtype=np.float32)
        self.knobs = np.zeros((batch_size,len(self.effect.knob_ranges)),dtype=np.float32)

        # given chunk size, batch size and overlap, figure out how long input audio can be
        self.signal_length = int( (chunk_size-overlap)*batch_size + overlap )
        self.num_nono_chunks = self.signal_length // self.chunk_size     # non-overlapping chunks for input
        self.input = np.zeros(self.signal_length,dtype=np.float32)   # 1D array for a batch's worth of input audio signal, spanning multiple chunks

        if path is not None:   # Use real audio files instead of synthetic generated data
            self.ra_gen = readaudio_generator(self.signal_length,  path=path, sr=self.sr,
                random_every=True)

    @autojit
    def get_x(self, chooser=None):  # gets a 1D array of input audio signal and breaks it into windows
        if self.ra_gen is None:
            for i in range(self.num_nono_chunks):
                ibn, iend = i * self.chunk_size, (i+1) * self.chunk_size
                self.input[ibn:iend ] = synth_input_sample(self.t, chooser)
        else:
            self.input = next(self.ra_gen)

        # break it up into overlapping windows
        return sliding_window(self.input, self.chunk_size, overlap=self.overlap)


    def gen_single_chunk(self, chooser=None, knobs=None, recyc_x=None):
        """create a single time-series of input and target output, all with the same knobs setting"""
        if chooser is None:
            chooser = np.random.choice([0,1,2,4,6,7])  # for compressor
            #chooser = np.random.choice([1,3,5,6,7])  # for echo

        if recyc_x is None:
            if self.ra_gen is None:
                x = synth_input_sample(self.t, chooser)
            else:
                x = next(self.ra_gen)
        else:
            x = recyc_x   # don't generate new x

        if knobs is None:
            knobs = random_ends(len(self.effect.knob_ranges))-0.5  # inputs to NN, zero-mean...except we emphasize the ends slightly

        y, x = self.effect.go(x, knobs)   # Apply the audio effect

        return x, y, knobs

    @autojit
    def gen_multiple_chunks(self, chooser=None, knobs=None, recyc_x=None):
        # get input audio stream
        self.x = self.get_x()  # windowed
        assert self.x.shape == (self.batch_size, self.chunk_size), "ShapeError: self.x.shape = "+str(self.x.shape)+" but (batch_size, chunk_size) = ("+str(self.batch_size)+","+str(self.chunk_size)+")"

        # generate knob settings for windows (which can be different)
        for line in range(self.batch_size):
            if knobs is None:
                knobs_i = random_ends(len(self.effect.knob_ranges))-0.5  # inputs to NN, zero-mean...except we emphasize the ends slightly
            else:
                knobs_i = knobs
            self.knobs[line,:] = knobs_i

        # and apply effect given that input and knobs (we could do this above with the knob settings, but I want to keep this 'clean')
        for line in range(self.batch_size):
            self.y[line,:], _ = self.effect.go(self.x[line], self.knobs[line])

        if self.effect.is_inverse:
            self.x, self.y = self.y, self.x

        return

    def new(self,chooser=None, knobs=None, recyc_x=False):
        # Generate new x, y, knobs BATCH
        individual_chunks = False
        if individual_chunks:
            # generate one chunk at a time
            for line in range(self.batch_size):  # generate separate chunks within each batch
                if recyc_x:
                    #self.x[line,:], self.y[line,:], self.knobs[line,:] = self.pool.apply_async(partial(self.gen_single, chooser, knobs, recyc_x=self.x[line,:])).get()
                    self.x[line,:], self.y[line,:], self.knobs[line,:] = self.gen_single_chunk(chooser, knobs=knobs, recyc_x=self.x[line,:])
                else:
                    #self.x[line,:], self.y[line,:], self.knobs[line,:] = self.pool.apply_async(partial(self.gen_single,chooser,knobs)).get()
                    self.x[line,:], self.y[line,:], self.knobs[line,:] = self.gen_single_chunk(chooser, knobs=knobs)
            #pool.close()
        else:
            self.gen_multiple_chunks(chooser, knobs=knobs, recyc_x=recyc_x)

        # Turn numpy data into torch/cuda data
        x_torch = torch.autograd.Variable(torch.from_numpy(self.x).to(self.device), requires_grad=self.requires_grad).float()
        y_torch = torch.autograd.Variable(torch.from_numpy(self.y).to(self.device), requires_grad=False).float()
        knobs_torch =  torch.autograd.Variable(torch.from_numpy(self.knobs).to(self.device), requires_grad=self.requires_grad).float()
        return x_torch, y_torch, knobs_torch

    def new_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.x = np.zeros((batch_size,self.chunk_size),dtype=np.float32)
        self.y = np.zeros((batch_size,self.chunk_size),dtype=np.float32)
        self.knobs = np.zeros((batch_size,len(self.effect.knob_ranges)),dtype=np.float32)



class AudioFileDataGenerator():  # different implementation of AudioDataGenerator, reads pre-made files
    """
    In this case we're just reading from (lots of files)
    """
    def __init__(self, chunk_size, sampling_freq, effect, batch_size=10, \
        requires_grad=True, device=torch.device("cuda:0"), overlap=0, path="./Train/", newfiles_every=10):

        super(AudioFileDataGenerator, self).__init__()
        global mp_shared_array1, mp_shared_array2, mp_shared_array3

        self.chunk_size = chunk_size
        self.sr = sampling_freq
        self.batch_size = batch_size
        self.requires_grad = requires_grad
        self.device = device
        self.path  = path
        self.overlap = overlap
        self.effect = effect  # The only reason we still need the effect defined (even though we're reading files) is to get the RANGES for the knobs

        # preallocate memory for the numpy version of the batch data the NN will use
        #  Our input audio will appear to be smaller than our X, array. We will later generate a 'view' of the input audio to have dimensions (batch_size,chunk_size)
        # see below.  self.x = np.zeros((batch_size,chunk_size),dtype=np.float32)  # input audio broken into chunks
        self.x = np.zeros((batch_size,chunk_size),dtype=np.float32)  # TODO: this is going to get overwritten. why allocate it?
        self.y = np.zeros((batch_size,chunk_size),dtype=np.float32)
        self.knobs = np.zeros((batch_size,len(self.effect.knob_ranges)),dtype=np.float32) # normalized values, -0.5 to 0.5



        # get a list of available files.  Note that knob settings are included to the target filenames
        self.input_filenames = sorted(glob.glob(path+'/'+'input_'+'*.wav'))
        self.target_filenames = sorted(glob.glob(path+'/'+'target_'+'*.wav'))
        print("path, len(self.input_filenames) = ", path, len(self.input_filenames))
        assert len(self.input_filenames) == len(self.target_filenames)   # TODO: One can image a scheme with multiple targets per input

        # Read in a TON of files, shuffle their parts
        self.num_load_files = 1000
        tmp_audio, sr = read_audio_file(self.input_filenames[0], sr=self.sr)
        self.x_file_buffer =  np.zeros((self.num_load_files, tmp_audio.shape[0]),dtype=np.float32)
        self.y_file_buffer =  np.zeros((self.num_load_files, tmp_audio.shape[0]),dtype=np.float32)
        self.knobs_buffer = np.zeros((self.num_load_files,len(self.effect.knob_ranges)),dtype=np.float32)

        # Allocate storage for shared multiprocessing arrays
        # clever use of ctypes for parallel loading of x and y arrays. Do NOT try to remove the tmp lines. Putting these on one line will run into a numpy garbage collection bug
        print("Allocating shared memory for data generator (this may take a while)")
        tmp1 = np.ctypeslib.as_ctypes(self.x_file_buffer)
        mp_shared_array1 = sharedctypes.RawArray(tmp1._type_, tmp1)
        tmp2 = np.ctypeslib.as_ctypes(self.y_file_buffer)
        mp_shared_array2 = sharedctypes.RawArray(tmp2._type_, tmp2)
        tmp3 = np.ctypeslib.as_ctypes(self.knobs_buffer)
        mp_shared_array3 = sharedctypes.RawArray(tmp3._type_, tmp3)
        self.fill_file_buffers()

    def read_one_new_file_pair(self):
        """
        this is designed to be run in parallel using a multiprocessing sharedctypes array
        """

        file_i = np.random.randint(0,high=len(self.input_filenames))
        print("file_i, file_in, file_target = ",file_i, self.input_filenames[file_i], self.target_filenames[file_i])
        audio_in, sr = read_audio_file(self.input_filenames[file_i], sr=self.sr)
        audio_targ, sr = read_audio_file(self.target_filenames[file_i], sr=self.sr)

        # parse knobs from target filename
        knobs_wc = self.parse_knob_string(self.target_filenames[file_i])
        # normalize knobs for nn usage
        kr = self.effect.knob_ranges
        knobs_nn = (knobs_wc - kr[:,0])/(kr[:,1]-kr[:,0]) - 0.5

        return audio_in, audio_targ, knobs_nn

    def fill_one_buffer_line(self,buffer_i):
        global mp_shared_array1, mp_shared_array2, mp_shared_array3
        # connect pointers to shared array
        tmp1 = np.ctypeslib.as_array(mp_shared_array1)
        tmp2 = np.ctypeslib.as_array(mp_shared_array2)
        tmp3 = np.ctypeslib.as_array(mp_shared_array3)

        audio_in, audio_targ, knobs_nn = self.read_one_new_file_pair()
        tmp1[buffer_i] = audio_in[0:self.x_file_buffer.shape[-1]]
        tmp2[buffer_i] = audio_targ[0:self.y_file_buffer.shape[-1]]
        tmp3[buffer_i] = knobs_nn
        return

    def fill_file_buffers(self):
        global mp_shared_array1, mp_shared_array2, mp_shared_array3
        indices = range(self.num_load_files)

        parallel = False
        if parallel:
            wrapper = partial(self.fill_one_buffer_line)
            num_procs = cpu_count()
            print("Reading",self.num_load_files,"files using",num_procs,"processes.")
            p = Pool(num_procs)
            result = p.map(wrapper, indices)
            self.x_file_buffer = np.ctypeslib.as_array(mp_shared_array1, shape=self.x_file_buffer.shape)
            self.y_file_buffer = np.ctypeslib.as_array(mp_shared_array2, shape=self.y_file_buffer.shape)
            self.knobs_buffer = np.ctypeslib.as_array(mp_shared_array3, shape=self.knobs_buffer.shape)
            p.close()
            p.join()
        else:
            for buffer_i in range(self.num_load_files):
                audio_in, audio_targ, knobs_nn = self.read_one_new_file_pair()
                self.x_file_buffer[buffer_i] = audio_in[0:self.x_file_buffer.shape[-1]]
                self.y_file_buffer[buffer_i] = audio_targ[0:self.y_file_buffer.shape[-1]]
                self.knobs_buffer[buffer_i] = knobs_nn


    def parse_knob_string(self, knob_str):  # given target filename, get knob settings
        """ By convention, we will use double-underscores before each knob setting.
            Nowhere else in the filename should double underscores appear.
        """
        #knob_str = 'target_9400_Compressor_4c__-10.95__3.428__0.005043__0.01308.wav'
        knob_list = knob_str.replace('.wav','').split('__')[1:]
        knobs = np.array([float(x) for x in knob_list])
        return knobs

    def grab_random_chunk(self):
        """ For now, this will be slow: open a random file, read it, pull out a small chunk of it
            But we'll do this in parallel
        """
        # pick a random line in the list of files
        buffer_i = np.random.randint(0,high=len(self.x_file_buffer.shape[0]))

        # ibgn & iend are set to grab a random chunk from the audio files
        ibgn = np.random.randint(0,self.x_file_buffer.shape[-1]-self.chunk_size)
        iend = ibgn + self.chunk_size

        x_chunk = self.x_file_buffer[buffer_i, ibgn:iend]
        y_chunk = self.y_file_buffer[buffer_i, ibgn:iend]
        knobs_nn = self.knobs_buffer[buffer_i]
        return x_chunk, y_chunk, knobs_nn


    def new(self):
        for i in range(self.batch_size):
                self.x[batch_i], self.y[batch_i], self.knobs[batch_i] = self.grab_random_chunk()

        # Turn numpy data into torch/cuda data
        x_torch = torch.autograd.Variable(torch.from_numpy(self.x).to(self.device), requires_grad=self.requires_grad).float()
        y_torch = torch.autograd.Variable(torch.from_numpy(self.y).to(self.device), requires_grad=False).float()
        knobs_torch =  torch.autograd.Variable(torch.from_numpy(self.knobs).to(self.device), requires_grad=self.requires_grad).float()
        return x_torch, y_torch, knobs_torch


    def new_batch_size(self, batch_size):  # sometimes we may want to change the batch size on the fly (currently unused)
        self.batch_size = batch_size
        self.x = np.zeros((batch_size,self.chunk_size),dtype=np.float32)
        self.y = np.zeros((batch_size,self.chunk_size),dtype=np.float32)
        self.knobs = np.zeros((batch_size,len(self.effect.knob_ranges)),dtype=np.float32)
'''
# EOF


__author__ = 'S.H. Hawley'

# imports
import numpy as np
import torch
from torch.autograd import Variable
import librosa
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
import scipy.signal as signal
from torch.utils.data.dataset import Dataset
from . import utils as st_utils
import os
import random

# optional utility for applying effects via sox
try:
    import pysox
except ImportError:
    no_pysox = True

#import matplotlib.pyplot as plt   # just for debugging
# Note: torchaudio is also a thing! requires sox. See http://pytorch.org/audio/, https://github.com/pytorch/audio




# reads a file. currently maps stereo to mono by throwing out the right channel
#   TODO: eventually we'd want to handle stereo somehow (e.g. for stereo effects)
def read_audio_file(filename, sr=44100):
    signal, sr = librosa.load(filename, sr=sr, mono=True) # convert to mono
    return signal, sr

def write_audio_file(filename, signal, sr=44100):
    librosa.output.write_wav(filename, signal, sr)
    return


# this generates a time-decaying sine wave, similar to plucking a string
def gen_pluck(length, t=None, amp=None, freq=None, decay=None, t0=0.0):
    if amp is None:
        amp = (0.6*np.random.random()+0.3)*np.random.choice([-1,1])
    if freq is None:
        freq = 300*np.random.random()
    if decay is None:
        decay = 10*np.random.random()
    if t is None:
        t = torch.linspace(0,1,length)
    pluck = amp * torch.exp(-decay * (t-t0) ) * torch.sin(freq* (t-t0))
    return pluck


# this operates using only one processor, and time-aligns one 'event' (which may be a 'pluck')
def ta_oneproc(input_sigs, target_sigs, chunk_size, event_len, num_events, strength, chunk_index):
    #  strength is a 0..1 'knob' that parameterizes the amount of time-shift applied: 0=no effect, 1=full ('on the grid')
    sig_length = chunk_size

    base_event = gen_pluck( int(event_len*1.5))            # 'base event' just the preceding event, to the right
    target_sigs[chunk_index, 0: base_event.size()[-1]] = base_event
    input_sigs[chunk_index, 0: base_event.size()[-1]] = base_event

    for eventnum in range(num_events):
        event = gen_pluck(event_len)

        # figure out where the erroneous input event should be, and where the target event should be
        grid_index = int(eventnum*event_len) + int(chunk_size/2)          # this is the target index value for "hard editing" / "on the grid"
        random_shift =  int ( (event_len/4)* (2*np.random.random()-1) )   # amount 'off' to place input event from the grid; the 1/4 is just an estimation

        input_index = grid_index + random_shift
        target_index =  int( strength*grid_index + (1.0-strength)*input_index )   # here's where the strength knob does its work

        itarget_bgn = target_index
        itarget_end = min(itarget_bgn + event_len, sig_length-1)
        target_sigs[chunk_index, itarget_bgn:itarget_end] = event[0 : itarget_end - itarget_bgn]

        # input: randomly shift it for input
        iinput_bgn = max(0, input_index )  # just don't let it go off the front end; TODO: this works but is sloppy;
        iinput_end = min(iinput_bgn + event_len, sig_length-1)
        input_sigs[chunk_index, iinput_bgn:iinput_end] = event[0: iinput_end - iinput_bgn ]

    return # note we don't have to return arrays because ThreadPool shares memory


# this runs in parallel, calling ta_oneproc many times to do multiple time-alignments
def gen_timealign_pairs(sig_length, num_events=1, parallel=True, strength=1.0):
    input_sigs = torch.zeros((num_chunks, chunk_size))
    target_sigs = input_sigs.clone()#  0.1*torch.randn((num_chunks, chunk_size))
    sig_length = chunk_size
    event_len = int( sig_length / (num_events+1) )
    chunk_indices = tuple( range(num_chunks) )

    if (parallel):
        pool = Pool()
        pool.map( partial(ta_oneproc, input_sigs, target_sigs, chunk_size, event_len, num_events, strength), chunk_indices)
        pool.close()
        pool.join()
    else:
        for chunk_index in chunk_indices:
            ta_oneproc(input_sigs, target_sigs, chunk_size, event_len, num_events, strength, chunk_index)
    return input_sigs, target_sigs


# generate pitch-shifted pair, on one processor
def psp_oneproc(input_sigs, target_sigs, fs, amp_fac, freq_fac, num_waves, chunk_index):
    sig_length = input_sigs.shape[1]
    for n in range(num_waves):
        # randomize the signal
        amp = 0.2*np.random.rand()    # stay bounded well below 1.0
        freq = 2 * np.pi * ( 400 + 400*np.random.rand() )

        # learn the adaptive filter for the following input -> target pair: different amp, freq & phase
        input_sigs[chunk_index]  +=           amp * torch.cos(           freq * torch.arange(sig_length) / fs)
        target_sigs[chunk_index] += amp_fac * amp * torch.sin(freq_fac * freq * torch.arange(sig_length) / fs)
    return  # note we don't have to return arrays because ThreadPool shares memory


# this generates groups of 'pitch shifted' pairs.  Not a true pitch-shifting effect, just something
#   I used temporarily until RenderMan VST host got upgraded to Python 3.6
# Runs in parallel, calls psp_oneproc many times
def gen_pitch_shifted_pairs(chunk_size, fs, amp_fac, freq_fac, num_waves, num_chunks, parallel=True):
        input_sigs = torch.zeros((num_chunks, chunk_size))
        target_sigs = torch.zeros((num_chunks, chunk_size))
        # generate them in parallel threads that all share the input_sigs and target_sigs arrays
        chunk_indices = tuple( range(num_chunks) )
        if (parallel):
            pool = Pool()
            pool.map( partial(psp_oneproc, input_sigs, target_sigs, fs, amp_fac, freq_fac, num_waves), chunk_indices)
            pool.close()
            pool.join()
        else:
            for chunk_index in chunk_indices:
                psp_oneproc(input_sigs, target_sigs, fs, amp_fac, freq_fac, num_waves, chunk_index)
        return input_sigs, target_sigs


# Generates various 'fake' audio wave forms -- synthetic data
def gen_input_sample(t, chooser=None):
    x = np.copy(t)*0.0
    if (chooser is None):
        chooser = np.random.randint(0,4)
    #chooser = 6
    #print("   make_input_signal: chooser = ",chooser)
    if (0 == chooser):
        amp = 0.4+0.4*np.random.random()
        freq = 30*np.random.random()    # sin, with random start & freq
        t0 = np.random.random()
        x = amp*np.sin(freq*(t-t0))
        x[np.where(t < t0)] = 0.0
        return x
    elif (1 == chooser):                # fixed sine wave
        freq = 5+150*np.random.random()
        amp = 0.4+0.4*np.random.random()
        global global_freq
        global_freq = freq
        return amp*np.sin(freq*t)
    elif (2 == chooser):                  # "pluck"
        amp0 = (0.6*np.random.random()+0.3)*np.random.choice([-1,1])
        t0 = (2*np.random.random()-1)*0.3
        decay = 8*np.random.random()
        freq = 6400*np.random.random()
        x = amp0*np.exp(-decay * (t-t0) ) * np.sin(freq* (t-t0))
        x[np.where(t < t0)] = 0   # without this, it grow exponentially 'to the left'
        return x
    elif (3 == chooser):                # ramp up then down
        height = (0.4*np.random.random()+0.2)*np.random.choice([-1,1])
        width = 0.3*np.random.random()/4   # half-width actually
        t0 = 2*width + 0.4*np.random.random() # make sure it fits
        x = height* ( 1 - np.abs(t-t0)/width )
        x[np.where(t < (t0-width))] = 0
        x[np.where(t > (t0+width))] = 0
        #x += 0.01
        return x
    elif (4 == chooser):                # 'box'
        height = (0.3*np.random.random()+0.2)*np.random.choice([-1,1])
        x = height*np.ones(t.shape[0])
        t1 = np.random.random()/2
        t2 = t1 + np.random.random()/2
        x[np.where(t<t1)] = 0.0
        x[np.where(t>t2)] = 0.0
        #x += 0.01
        return x
    elif (5 == chooser):                 # "bunch of spikes"
        n_spikes = 100
        for i in range(n_spikes):   # arbitrarily make a 'spike' somewhere, surrounded by silence
          loc = int(np.random.random()*len(t)-2)+1
          height = np.random.random()-0.5    # -0.5...0.5
          x[loc] = height
          x[loc+1] = height/2  # widen the spike a bit
          x[loc-1] = height/2
        x = x + 0.1*np.random.normal(0.0,scale=0.1,size=x.size)    # throw in noise
        return x
    elif (6 == chooser):                # white noise
        amp = 0.2+0.2*np.random.random()
        #amp = 2.0*np.random.random()
        x = amp*(2*np.random.random(t.shape[0])-1)
        return x
    else:
        x= 0.5*(gen_input_sample(t)+gen_input_sample(t)) # superposition of previous
        return x
    return np.copy(t)   # failsafe return just in case of typo above


# low pass filter
def lowpass(x, fc_fac=1.0):
    fc = fc_fac / len(x)
    b, a = signal.butter(1, fc, analog=False)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, x, zi=zi*x[0])
    return z


# this is a echo or delay effect  TODO: note the 'effect' string name for this is 'delay', not 'echo'. Pick one.
def echo(x, delay_samples=1487, echoes=2, ratio=0.6, dtype=np.float32):
    # ratio = redution ratio
    y = np.copy(x).astype(dtype)
    for i in range(echoes):
        ip1 = i+1
        delay_length = ip1 * delay_samples
        x_delayed = np.roll(x, delay_length)
        x_delayed[0:delay_length] = 0
        y += pow(ratio, ip1) * x_delayed
    return y


# simple compressor effect, code thanks to Eric Tarr @hackaudio
def compressor(x, thresh=-24, ratio=2, attack=2000, dtype=np.float32):
    fc = 1.0/(attack)               # this is like 1/attack time
    b, a = signal.butter(1, fc, analog=False)
    zi = signal.lfilter_zi(b, a)
    dB = 20*np.log10(np.abs(x) + 1e-8).astype(dtype)
    in_env, _ = signal.lfilter(b, a, dB, zi=zi*dB[0])  # input envelope calculation
    out_env = np.copy(in_env).astype(dtype)               # output envelope
    i = np.where(in_env >  thresh)          # compress where input env exceeds thresh
    out_env[i] = thresh + (in_env[i]-thresh)/ratio
    gain = np.power(10.0,(out_env-in_env)/10).astype(dtype)
    y = (np.copy(x) * gain).astype(dtype)
    return y



# Modified from https://bastibe.de/2012-11-02-real-time-signal-processing-in-python.html
class Limiter:
    def __init__(self, attack_coeff, release_coeff, delay, dtype=np.float32):
        self.delay_index = 0
        self.envelope = 0
        self.gain = 0.5
        self.delay = delay
        self.delay_line = np.zeros(delay, dtype=dtype)
        self.release_coeff = release_coeff
        self.attack_coeff = attack_coeff

    def limit(self, signal, threshold):
        out = np.copy(signal)
        for i in range(len(signal)):
            self.delay_line[self.delay_index] = signal[i]
            self.delay_index = (self.delay_index + 1) % self.delay

            # calculate an envelope of the signal
            self.envelope *= self.release_coeff
            self.envelope  = max(abs(signal[i]), self.envelope)

            # have self.gain go towards a desired limiter gain
            #print("i, self.envelope, threshold = ",i, self.envelope, threshold)
            if self.envelope > threshold:
                target_gain = (1+threshold-self.envelope)
            else:
                target_gain = 1.0
            self.gain = ( self.gain*self.attack_coeff +
                          target_gain*(1-self.attack_coeff) )

            # limit the delayed signal
            out[i] = self.delay_line[self.delay_index] * self.gain
        return out



def apply_sox_effect(signal, fxstr, sr=44100):
    """
    This writes signal to a .wav file, processes it sox to another file, loads that and returns it.

     signal:  a numpy list of numbers; the audio signal
     sr:      the sample rate in Hz, must be an integer
     fxstr:   a semicolon-separated string starting with the effect name followed by parameter values in order
              e.g., "lowpass;500" or "vol;18dB" or 'compand;0.3,0.8;-50', or if you're feeling ambitious,..
                    fxstr='ladspa;/usr/lib/ladspa/mbeq_1197.so;mbeq;-2;-3;-3;-6;-9;-9;-10;-8;-6;-5;-4;-3;-1;0;0;0'
              See the sox docs for more on effects: http://sox.sourceforge.net/sox.html#EFFECTS
              (Why semicolon-separated?  Because sox looks for both commas and colons!)
    """
    if (no_pysox is not None):
        raise ImportError("You don't have pysox installed.  Can't apply sox effect")
    inpath, outpath = 'in.wav', 'out.wav'
    librosa.output.write_wav(inpath, signal, sr)                     # write the input audio to a file
    tmp = fxstr.split(';')
    fxname = tmp[0]
    fxvals = [str.encode(x) for x in tmp[1:]]
    effectparams = [(fxname, fxvals),]
    print("effectparams=",effectparams)
    app = pysox.CSoxApp(inpath, outpath, effectparams=effectparams)   # apply the sox effect & get new file
    app.flow()

    out_signal, sr =  librosa.load(outpath, sr=sr)
    return out_signal[0:len(signal)]


def functions(x, f='id'):                # function to be learned
    """
       'functions' is a repository of various audio effects, which in some cases are
        literally just simple (time-independent) functions
    """

    if ('id' == f):
        return x 						# identity function
    elif ('x^2'==f):
        return x**2   					# given an x on [0,1], this should work
    elif ('clip' == f):
        return np.clip(x,-0.3,0.3)  	# hard limiter, clips signal
    elif ('sin' == f):
        return np.sin(4*x)              # just made this up
    elif ('dec_cos' == f):
        y = np.exp(-2 * x ) * np.cos(40* x) # decaying cosine
        return y
    elif ('wiggle' == f):
        y  = np.exp(-1*(x+0.7))*np.cos(20*x)			# decaying cos  wave
        y = y + np.exp(-40*(x-0.5)**2)					# plus  gaussian
        return y
    # low pass filter
    elif ('lpf' ==f):
        return lowpass(f)
    elif ('delay' == f) or ('echo' == f):
        return echo(x)
    elif ('comp' == f):
        return compressor(x)
    elif ('lim' == f):
        limiter = Limiter(.1,  0.5, 30)          # TODO: slow, always re-allocating delay line
        y = limiter.limit(x, 0.5)
        limiter = None                           # forced garbage collection.
        return y
    elif ('od' == f):   # really dumb/simple overdrive: a tanh!
        a = 3.0
        y = np.tanh(a*x)
        return y
    elif ('sox;' in f):                           # it's a sox effect, parse a semicolon-separated string
        fxstr = f.replace('sox;' , '')            # e.g. f = "sox,lowpass,500"; ignore the "sox;" part
        print(" Calling apply_sox_effect with fxstr =",fxstr,', len(x) =',len(x))
        y = apply_sox_effect(x, fxstr=fxstr)
        print("....and we're back, len(y) = ",len(y))
        return y
    else:
        raise ValueError("functions: error invalid type: "+f)



# reads audio from any number of audio files sitting in directory 'path'
# supplies a random window or chunk of lenth
def readaudio_generator(seq_size=8192*2, chunk_size=8192,  path='Train',
    basepath=os.path.expanduser('~')+'/datasets/signaltrain',
    random_every=True):
    # seq_size = amount of audio samples to supply from file
    # basepath = directory containing Train, Val, and Test directories
    # path = audio files for dataset  (can be Train, Val or test)
    # random_every = get a random window every time next is called, or step sequentially through file
    path = basepath+'/'+path+'/'
    files = os.listdir(path)
    sr = 44100
    read_new_file = True
    start = -seq_size
    while True:
        if read_new_file:
            filename = path+np.random.choice(files)  # pick a random audio file in the directory
            print("Reading new data from "+filename+" ")
            data, sr = read_audio_file(filename, sr=sr)
            read_new_file=False   # don't keep switching files  everytime generator is called


        if (random_every): # grab a random window of the signal
            start = np.random.randint(0,data.shape[0]-seq_size)
        else:
            start += seq_size
        xraw = data[start:start+seq_size]   # the newaxis just gives us a [1,] on front
        # Note: any 'windowing' happens after the effects are applied, later
        rc = ( yield xraw )         # rc is set by generator's send() method.
        if isinstance(rc, bool):    # can set read_new by calling send(True)
            read_new_file = rc




def samplecat_ta(path=os.path.expanduser('~')+'/SampleCat/DrumSamples_large/',
    sr=44100, bpm=90, seq_size=24900*1000, shuffle=True):
    """
    Concatenates audio clips/samples (e.g. drums), generates time-aligned and
    and randomly-shifted variants, as well as a click track

    Inputs:
        path:     Directory where .WAV sample files are located
        sr:       Sample rate in Hz
        bpm:      Beats per minute for time-alignment and click generation
        seq_size: Length in samples of generated audio signal
        shuffle:  Whether or not to randomly permute the order of sample files

    Outputs:
        final_clip_randomized: Audio sequence of many samples, randomly shifted
        final_clip:            Audio sequence of same samples, 'on the grid'
        click:                 Audio click track corresponding to the 'grid'

    Author: B.L. Colburn, benjamin.colburn@pop.belmont.edu
    """
    print("Initializing samplecat_ta:",end="")
    click_delay = 60/bpm    # in seconds
    sample_delay = librosa.core.time_to_samples(click_delay, sr=sr)
    try:
        files = os.listdir(path)
    except:
        print("Error! Can't find audio samples in",path)
        return None

    final_clip = np.zeros((seq_size,),dtype=np.float32)
    final_clip_randomized = np.zeros((seq_size+sample_delay,),dtype=np.float32)

    paths = []
    for f in files:
        if f.endswith(".WAV"):
            paths.append(path + "/" +f)
    print("",len(paths),"audio clips available in",path)

    while True:   # can use this as a generator, call inf. many times
        if shuffle:
            print("samplecat_ta: shuffling order of clips...")
            random.shuffle(paths)
            shuffle = False    # turn it off so it only shuffles when asked to

        # fill up final_clip & final_clip_randomized with audio clips/samples
        i, current_size = 0, 0
        while current_size<final_clip.shape[0] and i < len(paths):
            if i != 0:
                r_shift = int(sample_delay/4 * (2*np.random.rand()-1))
            else:
                r_shift = int(sample_delay/int(random.randrange(4,7))) # first clip is never early, only late
            start   = i * sample_delay
            start_s = i * sample_delay + r_shift
            end   = min(start + sample_delay,  final_clip.shape[0])
            end_s = min(start_s + sample_delay,  final_clip.shape[0])
            clip, _ = librosa.core.load(paths[i], mono=True, sr=sr, duration=4.0)  # ", _" means we throw out the sr returned by load
            clip = librosa.util.fix_length(clip, sample_delay)
            final_clip[start:end] = clip[0:end-start]
            final_clip_randomized[start_s:end_s] = clip[0:end_s-start_s]
            i=i+1
            current_size = current_size + clip.shape[0]

        # produce the click track
        duration = librosa.core.get_duration(final_clip, sr=sr)
        num_clicks = int( np.ceil(duration / click_delay) )   # ceil rounds up
        times = np.arange(num_clicks) * click_delay
        click = librosa.core.clicks(times=times, sr=sr, click_duration=click_delay, click_freq=200, length=final_clip.shape[0])

        #? SHH: why is this here?
        #final_clip = np.trim_zeros(final_clip,'b')
        #final_clip_randomized = np.trim_zeros(final_clip_randomized,'b')

        # How about instead of that: make three audio arrays the same length
        min_length = min( min( final_clip.shape[0], final_clip_randomized.shape[0]), click.shape[0])

        # return values from the generator; shuffle can be (re)set set via send() method
        shuffle = ( yield final_clip_randomized[0:min_length], final_clip[0:min_length], click[0:min_length] )

        final_clip *= 0   # when generator is called again, start here & clear variables
        final_clip_randomized *= 0




def gen_audio(seq_size=8192*2, chunk_size=8192,  path='Train',
    basepath=os.path.expanduser('~')+'/datasets/signaltrain',
    random_every=True, effect='ta'):
    """
    General wrapper for audio generators, that sets up windowed 'stacks'
    and converts to pytorch tensors
    """

    if effect != 'ta':
        signal_gen = readaudio_generator(seq_size, chunk_size, path, basepath, random_every)
    else:
        signal_gen = samplecat_ta(seq_size=seq_size)

    while True:
        if effect != 'ta':
            X = next(signal_gen)
            Y = functions(X, f=effect)
        else:
            X, Y, click = next(signal_gen)
        X = torch.from_numpy(st_utils.chopnstack(X, chunk_size)).unsqueeze(0)
        Y = torch.from_numpy(st_utils.chopnstack(Y, chunk_size)).unsqueeze(0)
        X.requires_grad, Y.requires_grad = False, False
        rc = (yield X, Y)
        if isinstance(rc, bool):    # our send method can pass data through
            signal_gen.send(rc)



'''
def gen_synth_audio(sig_length, batch_size, device, chunk_size=8192, effect='ta', input_var=None, \
    target_var=None, mu_law=False, x_grad=True, fs=44100):
    dtype=np.float32
    """
    ---------------------------------------------------------------------------------------
       gen__synth_audio is a routine that provides synthetic audio data.
       TODO: should probably try adding mu-law companding to see if that helps with SNR
       Inputs:
             sig_length:  is actually the totally length (in samples) of the the entire dataset,
                         conceived as if it were just one file.
             device:     the pytorch device where the computation is performed
             chunk_size: chop up the signal into chunks of this length.
             effect:     string corresponding to a name of an audio effect
                         Some of these just generate generic input & apply the effect to that,
                         other 'effects' may involve generating input & output together
            input_var & target_var: These *can* be passed in so that memory gets freed up
                                    before generating new data. (It'd be nice to be able to
                                    do everything 'in place', but w/ PyTorch vs. numpy, and
                                    GPU vs CPU, this can get complicated.)
            mu_law:      Set to true to apply mu-law encoding to input & target audio
            x_grad:      Normally the signal_var generated gets a gradient, but by setting this
                         to False you can save memory
            fs:          sample rate
    ---------------------------------------------------------------------------------------
    """
    # Free up pytorch tensor memory usage before generating new data
    if (input_var is not None) and (target_var is not None):
        del input_var, target_var

    overlap = 0 #int(chunk_size*0.1)    # this by how many elements the chunks (windows) will overlap
    unique_per_chunk = chunk_size - 2*overlap

    num_chunks = int(np.ceil(sig_length / unique_per_chunk))

    sig_length = num_chunks * chunk_size   # if requested size isn't integer multiple of chunk_size, we zero-pad

    input_sig = np.zeros( (batch_size, sig_length) ).astype(dtype)       # allocate storage
    target_sig = np.zeros( (batch_size, sig_length) ).astype(dtype)       # allocate storage

    for bi in range(batch_size):  #  "bi" = "batch index"
        # define new
        if False and ('ps' == effect):    # pitch shift  TODO: this is broken for now
            fs = 44100.
            num_waves = 20
            amp_fac = 0.43
            freq_fac = 0.31
            input_stack, target_stack = gen_pitch_shifted_pairs(chunk_size, fs, amp_fac, freq_fac, num_waves, num_chunks)
        else:      # other effects, where target can be generated from input instead of both together
            # Generate input signal via a series of samaple 'clips'
            clip_size = 22050      # about     0.5 seconds
            num_sample_clips = int(sig_length / clip_size)
            t = np.linspace(0,1,num=clip_size).astype(dtype)
            sample_type = 2 # 'pluck'
            for i in range(num_sample_clips):        #  go through each section of input and assign a waveform
                clip = gen_input_sample(t,chooser=sample_type)            # the audio clip to add

                start_ind = i * clip_size                                 # 'on the grid'
                end_ind = min( start_ind + clip_size, sig_length )        # just don't go off the end of the buffer
                this_clip_len = end_ind - start_ind       # length of this clip
                target_sig[bi, start_ind:start_ind+this_clip_len] = clip[0:this_clip_len]  # this will get overwritten if we're not doing time alignment

                r_shift = int(0.3*clip_size* (2*np.random.rand()-1))          # random_shift, early or late
                start_ind = start_ind + r_shift                               # 'on the grid'
                if start_ind < 0:
                    clip_start_ind = -start_ind
                    clip_end_ind = start_ind + clip_size
                    input_sig[bi, 0:clip_end_ind] = clip[0:clip_end_ind]  #     input is like target but shifted off the grid
                else:
                    end_ind = min( start_ind + clip_size, sig_length )        # just don't go off the end of the buffer
                    this_clip_len = end_ind - start_ind       # length of this clip
                    input_sig[bi, start_ind:start_ind+this_clip_len] = clip[0:this_clip_len]  # input is like target but shifted off the grid


            if ('delay' == effect):
                input_sig *= 0.5    # for delay, just make it even smaller to avoid any clipping that may occur

            # Apply the effect, whatever it is except time alignment
            if ('ta' != effect):
                target_sig[bi] = functions(input_sig[bi], f=effect)

            if mu_law:
                input_sig = encode_mu_law(input_sig)
                target_sig = encode_mu_law(target_sig)
                dtype = np.long

    # chop up the input & target signal(s)
    #input_sig.shape = (batch_size, num_chunks, chunk_size)
    #target_sig.shape = (batch_size, num_chunks, chunk_size)


    input_stack =  chopnstack(input_sig, chunk_size=chunk_size, overlap=overlap)
    target_stack = chopnstack(target_sig, chunk_size=chunk_size, overlap=overlap)

    input_var = Variable(torch.from_numpy(input_stack), requires_grad=x_grad).to(device)
    target_var = Variable(torch.from_numpy(target_stack), requires_grad=False).to(device)

    return input_var, target_var
'''




# mu laws from https://gist.github.com/lirnli/4282fcdfb383bb160cacf41d8c783c70
def encode_mu_law(x, mu=256):
    mu = mu-1
    fx = np.sign(x)*np.log(1+mu*np.abs(x))/np.log(1+mu)
    return np.floor((fx+1)/2*mu+0.5).astype(np.long)

def decode_mu_law(y, mu=256):
    mu = mu-1
    fx = (y-0.5)/mu*2-1
    x = np.sign(fx)/mu*((1+mu)**np.abs(fx)-1)
    return x




# EOF

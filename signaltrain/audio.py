
# -*- coding: utf-8 -*-
__author__ = 'S.H. Hawley'

# imports
import numpy as np
import scipy.signal as scipy_signal
import torch
#import torchaudio
import librosa
from numba import autojit, njit, jit   # Note: nopython version gives symbol errors when used w/ Jupyter Notebook, so using autojit instead
import os
import sys
import glob
#import io_methods
from scipy.io import wavfile
import configparser  # for reading file-based effect datasets
import warnings

def random_ends(size=1): # probabilty dist. that emphasizes boundaries
    return np.random.beta(0.8,0.8,size=size)

@autojit
def sliding_window(x, size, overlap=0):
    """
    Stacks 1D array into a series of sliding windows with a certain amount of overlaps.
    This is fast because it generates a "view" rather than creating a new array.
    -->Unless the windows don't divide evenly, in which case we pad with zeros to get an even coverage
    Inputs:
       x:  the 1D array to be windowed
       size:  the width of each window
       overlap = amount of "lookback" (in samples), when predicting the next set of values
    Example:
        x = np.arange(10)
        print(sliding_window(x, 5, overlap=2))
         [[0 1 2 3 4]
         [3 4 5 6 7]
         [6 7 8 9 0]]
    Source: from last answer to https://stackoverflow.com/questions/4923617/efficient-numpy-2d-array-construction-from-1d-array
    """
    step = size - overlap # amount of non-overlapped values per window
    remainder = (x.shape[-1]-size) % step   # see if array will divide up evenly
    if remainder != 0:
        x = np.pad(x, (0,step-remainder), mode='constant') # pad end with zeros until it does. note this changes the size of x

    nwin = (x.shape[-1]-size)//step + 1  # this truncates any leftover rows, rather than padding with zeros
    shape = x.shape[:-1] + (nwin, size)
    strides = x.strides[:-1] + (step*x.strides[-1], x.strides[-1])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False) # writeable=False is to avoid memory corruption. better safe than sorry!

'''
def undo_siding_window_view(x, overlap):
    """
    Undoes the the sliding window view: Returns 1-D shape of length equal to nearest
        multiple of window, minus overlap
    NOTE: This operates only on a view.  It does not remove windows from copies or new arrays
    """
    shape = [(overlap + len(x[:,overlap:])//overlap)*overlap - overlap + 1]
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=[x.strides[-1]], writeable=False)
'''
def undo_sliding_window(x, overlap, flatsize=None):
    """
    This works in general, i.e. for views and for copies of arrays.
    NOTE: only undoes padding that might have occurred if flatsize != None.
    """
    if overlap != 0:
        xnew =  np.concatenate( (x[0,0:overlap],  x[:,overlap:].flatten() ) )
        if flatsize is not None:
          return xnew[0:flatsize]
        else:
          return xnew
    else:
        return x


#--- List of test signals:
def pinknoise(N):
    """
    Generates 1/f noise
      N = length of array to generate
    """
    N_f = N //2 + 1
    noise = 2*np.random.random(N_f)-1
    s = np.sqrt(np.arange(len(noise)) + 1.)  # +1 avoids dividing by zero
    y = (np.fft.irfft(noise / s)).real
    return y/np.max(np.abs(y))  # normalize

def randsine(t, randfunc=np.random.rand, amp_range=[0.2,0.9], freq_range=[5,150], n_tones=None, t0_fac=None):
    y = np.zeros(t.shape[0])
    if n_tones is None: n_tones=np.random.randint(1,3)
    for i in range(n_tones):
        amp = amp_range[0] + (amp_range[1]-amp_range[0])*randfunc()
        freq = freq_range[0] + (freq_range[1]-freq_range[0])*randfunc()
        t0 = randfunc() * t[-1] if t0_fac is None else t0_fac*t[-1]
        y += amp*np.cos(freq*(t-t0))
    return y

def box(t, randfunc=np.random.rand, t0_fac=None, delta=None):
    """
    classic "box"-shaped step response
    t0_fac: specifies faction of total length at which to start at (otherwise random)
    """
    height_bgn, height_mid, height_end = 0.15*randfunc(), 0.35*randfunc()+0.6, 0.2*randfunc()+0.1
    maxi = len(t)
    delta = np.random.choice([0, np.random.randint(100) ]) if delta is None else delta # maybe slope the sides ; delta=0 is an immediate step response
    i_up = delta+int( 0.3*randfunc() * maxi) if t0_fac is None else int(t0_fac*maxi)
    i_dn = min( i_up + int( (0.3+0.35*randfunc())*maxi), maxi-delta-1)   # time for jumping back down
    x = height_end*np.ones(t.shape[0]).astype(t.dtype, copy=False)  # unit amplitude
    x[0:i_up-1] = height_bgn
    x[i_up:i_dn] = height_mid    # the "flat top" middle area of the box
    if delta > 0:
        x[i_up-delta:i_up+delta] = height_bgn + (height_mid-height_bgn)*(np.arange(2*delta))/2/delta
        x[i_dn-delta:i_dn+delta] = height_mid - (height_mid-height_end)*(np.arange(2*delta))/2/delta
    return x

def expdecay(t, randfunc=np.random.rand, t0_fac=None, high_fac=None, low_fac=None):
    """generic exponential decay envelope; called by other routines (below)
       t0_fac is fraction of final time at which to start
    """
    t0 = 0.35*randfunc()*t[-1] if t0_fac is None else t0_fac*t[-1]
    height_high = 0.35*randfunc() + 0.6 if high_fac is None else high_fac
    height_low = 0.1*randfunc()+0.1 if low_fac is None else low_fac
    decay = 12*randfunc()
    x = np.exp(-decay * (t-t0)) * height_high   # decaying envelope
    x[np.where(t < t0)] = height_low   # without this, it grow exponentially 'to the left'
    return x

def pluck(t, randfunc=np.random.rand, freq_range=[50,6400], n_tones=None, t0_fac=None):
    y = np.zeros(t.shape[0])
    """ supposed to be like a plucked string; but with a few random tones as well"""
    if n_tones is None: n_tones=np.random.randint(1,4)
    for i in range(n_tones):
        amp0 = (0.45 * randfunc() + 0.5) * np.random.choice([-1, 1])
        t0 = (2. * randfunc()-1)*0.3 * t[-1] if t0_fac is None else t0_fac*t[-1] # for phase
        freq = freq_range[0] + (freq_range[1]-freq_range[0])*randfunc()
        y += amp0*np.sin(freq * (t-t0))
    return y * expdecay(t, t0_fac=t0_fac)

def ampexpstepup(t, randfunc=np.random.rand, freq=None, freq_range=[400,5000], start_dB=-40):
    """ sine wave with exponentially increase amplitude
    cf. Figure 3 of AES Conf Paper 6849: "Parameter Estimation of Dynamic Range Compressors: Models, Procedures and Test Signals"
    http://www.aes.org/e-lib/browse.cfm?elib=13653
    ...except we typically won't do a signal that long (theirs is ~120 seconds)
    ...Looks lik they stey by 1dB for about 50 steps across the clip
    """
    env_dB = np.floor( np.linspace(start_dB, 0, num=len(t)) ) # envelope in integer steps from start_dB to 0dB
    env = np.power(10.0, env_dB/10)                      # envelope in float values
    if freq is None:  #  Otherwise, the user has specified a frequency in Hz
        freq = freq_range[0] + (freq_range[1]-freq_range[0])*randfunc() # pick a freq
    return  env * np.sin(freq * t)

def sweep(t, randfunc=np.random.rand, freq_range=[20,20000], amp_too=False):
    """exponential frequency sweep
    """
    tmax = t[-1]
    lnfr = np.log(freq_range[1]/freq_range[0])  # ln of frequency ratio
    amp = 0.9*randfunc()
    y =  amp* np.sin( 20 *2*np.pi*tmax/lnfr * (np.exp(t/tmax*lnfr)-1) )
    if amp_too:         # exponentially increase the amplitude as well
        y *= np.exp(lnfr*t/tmax)
    return y

def spikes(t, n_spikes=50, randfunc=np.random.rand):  # "bunch of random spikes"
    x = np.zeros(t.shape[0])
    for i in range(n_spikes):   # arbitrarily make a 'spike' somewhere, surrounded by silence
      loc = int( int(randfunc()*len(t)-2)+1* t[-1] )
      height = (2*randfunc()-1)*0.7    # -0.7...0.7
      x[loc] = height
      x[loc+1] = height/2  # widen the spike a bit
      x[loc-1] = height/2

    amp_n = 0.1*randfunc()
    x = x + amp_n*np.random.normal(size=t.shape[0])    # throw in noise
    return x

def triangle(t, randfunc=np.random.rand, t0_fac=None): # ramp up then down
    height = (0.4 * randfunc() + 0.4) * np.random.choice([-1,1])
    width = randfunc()/4 * t[-1]     # half-width actually
    t0 = 2*width + 0.4 * randfunc()*t[-1] if t0_fac is None else t0_fac*t[-1]
    x = height * (1 - np.abs(t-t0)/width)
    x[np.where(t < (t0-width))] = 0
    x[np.where(t > (t0+width))] = 0
    amp_n = (0.1*randfunc()+0.02)   # add noise
    return x + amp_n*pinknoise(t.shape[0])


# Prelude to read_audio_file
# Tried lots of ways of doing this.. most are slow.
#signal, rate = librosa.load(filename, sr=sr, mono=True, res_type='kaiser_fast') # Librosa's reader is incredibly slow. do not use
#signal, rate = torchaudio.load(filename)#, normalization=True)   # Torchaudio's reader is pretty fast but normalization is a problem
#signal = signal.numpy().flatten()
#reader = io_methods.AudioIO   # Stylios' file reader. Haven't gotten it working yet
#signal, rate = reader.audioRead(filename, mono=True)
#signal, rate = sf.read('existing_file.wav')
def read_audio_file(filename, sr=44100, mono=True, norm=False, device='cpu', dtype=np.float32):
    """
    Generic wrapper for reading an audio file.
    Different libraries offer different speeds for this, so this routine is the
    'catch-all' for whatever read routine happens to work best

    Tries a fast method via scipy first, reverts to slower librosa when necessary.
    """
    # first try to read via scipy, because it's fast
    scipy_ok = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error")    # scipy throws warnings which should be errors
        try:
            out_sr, signal = wavfile.read(filename)
            scipy_ok = True
        except wavfile.WavFileWarning:
            print("read_audio_file: Warning raised by scipy. ",end="")

    if scipy_ok:
        if mono and (len(signal.shape) > 1):     # convert to mono
            signal = signal[:,0]

        if isinstance(signal[0], np.int16):      # convert from ints to floats if necessary
            signal = np.array(signal/32767.0, dtype=dtype)   # change from [-32767..32767] to [-1..1]

        if out_sr != int(sr):
            print(f"read_audio_file: Got sample rate of {rate} Hz instead of {sr} Hz requested. Resampling.")
            signal = librosa.resample(signal, rate*1.0, sr*1.0, res_type='kaiser_fast')

    else:                                         # try librosa; it's slower but general
        print("Trying librosa.")
        signal, out_sr = librosa.core.load(filename, mono=mono, sr=sr, res_type='kaiser_fast')

    if signal.dtype != dtype:
        signal = signal.astype(dtype, copy=False)

    if norm:
        absmax = np.max(np.abs(signal))
        signal = signal/absmax if absmax > 0 else signal

    return signal, out_sr


def write_audio_file(filename, data, sr=44100):
    wavfile.write(filename, sr, data)
    #librosa.output.write_wav(filename, data, sr)
    #torchaudio.save(filename, torch.Tensor(data).unsqueeze(1), sr)
    return

def readaudio_generator(seq_size,  path=os.path.expanduser('~')+'/datasets/signaltrain/Val', sr=44100,
    random_every=True, mono=True, norm=False):
    """
    reads audio from any number of audio files sitting in directory 'path'
    supplies a window of length "seconds". If random_every=True, this window will be randomly chosen
    """
    # seq_size = amount of audio samples to supply from file
    # basepath = directory containing Train, Val, and Test directories
    # path = audio files for dataset  (can be Train, Val or test)
    # random_every = get a random window every time next is called, or step sequentially through file
    files = glob.glob(path+"*.wav")
    read_new_file = True
    start = -seq_size
    while True:
        if read_new_file:
            filename = path+'/'+np.random.choice(files)  # pick a random audio file in the directory
            #print("Reading new data from "+filename+" ")
            data, sr = read_audio_file(filename, sr=sr, mono=mono, norm=norm)
            read_new_file=False   # don't keep switching files  everytime generator is called


        if (random_every): # grab a random window of the signal
            start = np.random.randint(0,data.shape[0]-seq_size)
        else:
            start += seq_size
        xraw = data[start:start+seq_size]   # the newaxis just gives us a [1,] on front
        # Note: any 'windowing' happens after the effects are applied, later
        rc = ( yield xraw )         # rc is set by generator's send() method.  YIELD here is the output
        if isinstance(rc, bool):    # can set read_new by calling send(True)
            read_new_file = rc


def synth_input_sample(t, chooser=None, randfunc=np.random.rand, t0_fac=None):
    """
    Synthesizes one instance from various 'fake' audio wave forms -- synthetic data
    """
    if chooser is None:
        chooser = np.random.randint(0, 11)

    if 0 == chooser:                     # sine, with random phase, amp & freq
        return randsine(t, t0_fac=t0_fac)
    elif 1 == chooser:                  # noisy sine
        return randsine(t,t0_fac=t0_fac) + 0.2*np.random.rand()*pinknoise(t.shape[0]) + 0.2*np.random.rand()*(2*np.random.rand(t.shape[0])-1)
    elif 2 == chooser:                    #  "pluck", decaying sine wave
        return pluck(t,t0_fac=t0_fac)
    elif 3 == chooser:                   # ramp up then down
        return triangle(t,t0_fac=t0_fac)
    elif (4 == chooser):                # 'box'
        return box(t,t0_fac=t0_fac)
    elif 5 == chooser:                 # "bunch of spikes"
        return spikes(t)
    elif 6 == chooser:                # noisy box
        return box(t,t0_fac=t0_fac) * (2*np.random.rand(t.shape[0])-1) # don't use pinknoise here
    elif 7 == chooser:                # noisy 'pluck'
        amp_n = (0.3*randfunc()+0.1)
        return pluck(t,t0_fac=t0_fac) + amp_n*pinknoise(t.shape[0])
    elif 8 == chooser:
        return ampexpstepup(t, start_dB=-30) # increasing amplitude-steps of sine wave
    elif 9 == chooser:                       # freq sweep
        f_low, f_high  = np.random.randint(20,1000), np.random.randint(1000,20000)
        amp_too = np.random.choice([False,False,True])
        return sweep(t, freq_range=[f_low, f_high], amp_too=amp_too)
    elif 10 == chooser:                     # box plus noise
        return st.audio.box(t) + 0.2*np.random.rand()*(2*np.random.rand(t.shape[0])-1) + 0.2*np.random.rand()*pinknoise(t.shape[0])
    elif 11 == chooser:                  # just noise
        amp_n = (0.6*randfunc()+0.2)
        return amp_n*pinknoise(t.shape[0])
    else:
        return 0.5*(synth_input_sample(t)+synth_input_sample(t)) # superposition of the above
#---- End test signals


#---- Effects
@autojit
def compressor(x, thresh=-24, ratio=2, attackrel=0.045, sr=44100.0, dtype=np.float32):
    """
    simple compressor effect, code thanks to Eric Tarr @hackaudio
    Inputs:
       x:        the input waveform
       thresh:   threshold in dB
       ratio:    compression ratio
       attackrel:   attack & release time in seconds
       sr:       sample rate
    """
    attack = attackrel * sr  # convert to samples
    fc = 1.0/float(attack)     # this is like 1/attack time
    b, a = scipy_signal.butter(1, fc, analog=False, output='ba')
    zi = scipy_signal.lfilter_zi(b, a)

    dB = 20. * np.log10(np.abs(x) + 1e-6)
    in_env, _ = scipy_signal.lfilter(b, a, dB, zi=zi*dB[0])  # input envelope calculation
    out_env = np.copy(in_env)              # output envelope
    i = np.where(in_env >  thresh)          # compress where input env exceeds thresh
    out_env[i] = thresh + (in_env[i]-thresh)/ratio
    gain = np.power(10.0,(out_env-in_env)/20)
    y = x * gain
    return y

@jit(nopython=True)
def my_clip_min(x, clip_min):  # does the work of np.clip(), which numba doesn't support yet
    # TODO: keep an eye on Numba PR https://github.com/numba/numba/pull/3468 that fixes this
    inds = np.where(x < clip_min)
    x[inds] = clip_min
    return x

@jit(nopython=True)
def compressor_4controls(x, thresh=-24.0, ratio=2.0, attackTime=0.01, releaseTime=0.01, sr=44100.0):
    """
    Thanks to Eric Tarr for MATLAB code for this, p. 428 of his Hack Audio book.  Used with permission.
    Our mods for Python:
        Minimized the for loop, removed dummy variables, and invoked numba @jit to make this "fast"
    Inputs:
      x: input signal
      sr: sample rate in Hz
      thresh: threhold in dB
      ratio: ratio (should be >=1 , i.e. ratio:1)
      attackTime, releaseTime: in seconds
    """
    N = len(x)
    dtype = x.dtype
    y = np.zeros(N, dtype=dtype)
    lin_A = np.zeros(N, dtype=dtype)  # functions as gain

    # Initialize separate attack and release times
    alphaA = np.exp(-np.log(9)/(sr * attackTime))#.astype(dtype,copy=False)  numba doesn't support astype
    alphaR = np.exp(-np.log(9)/(sr * releaseTime))#.astype(dtype,copy=False)

    # Turn the input signal into a uni-polar signal on the dB scale
    x_uni = np.abs(x)
    x_dB = 20*np.log10(x_uni + 1e-8)  # x_uni casts type

    # Ensure there are no values of negative infinity
    #x_dB = np.clip(x_dB, -96, None)   # Numba doesn't yet support np.clip but we can write our own
    x_dB = my_clip_min(x_dB, -96)

    # Static Characteristics
    gainChange_dB = np.zeros(x_dB.shape[0], dtype=dtype)
    i = np.where(x_dB > thresh)
    gainChange_dB[i] =  thresh + (x_dB[i] - thresh)/ratio - x_dB[i] # Perform Downwards Compression

    for n in range(1, N):  # this loop is slow but not vectorizable due to its cumulative, sequential nature. @autojit makes it fast(er).
        # smooth over the gainChange
        if gainChange_dB[n] < lin_A[n-1]:
            lin_A[n] = ((1-alphaA)*gainChange_dB[n]) +(alphaA*lin_A[n-1]) # attack mode
        else:
            lin_A[n] = ((1-alphaR)*gainChange_dB[n]) +(alphaR*lin_A[n-1]) # release

    lin_A = np.power(10.0,(lin_A/20))  # Convert to linear amplitude scalar; i.e. map from dB to amplitude

    y = lin_A * x    # Apply linear amplitude to input sample

    return y


 # this is a echo or delay function
def echo(x, delay_samples=1487, ratio=0.6, echoes=1):
    # ratio = redution ratio
    dtype = x.dtype
    y = np.copy(x)
    for i in range(int(np.round(echoes))):   # note 'echoes' is a 'switch'; does not vary continuously
        ip1 = i+1       # literally "i plus 1"
        delay_length = ip1 * delay_samples
        delay_length_int = int(np.floor(delay_length))
        # the following is an attempt to make the delay continuously differentiable
        diff = delay_length - delay_length_int
        x_delayed = ( (1-diff)*np.pad(x,(delay_length_int,0),mode='constant')[0:-delay_length_int] #shift and pad with zeros
                        + diff*np.pad(x,(delay_length_int+1,0),mode='constant')[0:-(delay_length_int+1)])
        y += pow(ratio, ip1) * x_delayed
    return y




# Classes for Effects. First is the generic/main class. All others are subclass of this
class Effect():
    """Generic effect super-class
       sub-classed Effects should also define a 'go_wc()' method to execute the actual effect
       Network will call go() with normalized knob values, which then will call go_wc()
       The go_wc() method should return two value: y, x   where y is target output and x is input signal
    """
    def __init__(self, sr=44100.0, dtype=np.float32):
        self.name = 'Generic Effect'
        self.knob_names = ['knob']
        self.knob_ranges = np.array([[0,1]], dtype=dtype)  # min,max world coordinate values for "all the way counterclockwise" and "all the way clockwise"
        self.sr = sr
        self.is_inverse = False  # Does this effect perform an 'inverse problem' by reversing x & y at the end?

    def knobs_wc(self, knobs_nn):   # convert knob vals from [-.5,.5] to "world coordinates" used by effect functions
        return (self.knob_ranges[:,0] + (knobs_nn+0.5)*(self.knob_ranges[:,1]-self.knob_ranges[:,0])).tolist()

    def info(self):  # Print some information about the effect
        assert len(self.knob_names)==len(self.knob_ranges)
        print(f'Effect: {self.name}.  Knobs:')
        for i in range(len(self.knob_names)):
            print(f'                            {self.knob_names[i]}: {self.knob_ranges[i][0]} to {self.knob_ranges[i][1]}')
        if self.is_inverse:
            print("                            <<<< INVERSE EFFECT <<<<")
    # Effects should also define a 'go_wc' method which executes the effect, mapping input and knobs_nn to output y, x
    #   We return x as well as y, because some effects may reverse x & y (e.g. denoiser)
    def go_wc(self, x, knobs_wc):
        raise Exception("This effect's go_wc() is undefined")

    # this is the 'main' interface typically called during training & inference, using normalized knob values [-.5,.5]
    def go(self, x, knobs_nn, **kwargs):
        knobs_w = self.knobs_wc(knobs_nn)
        return self.go_wc(x, knobs_w, **kwargs)


# The following are some "plugins", which call functions that have been defined above
class Compressor(Effect):
    def __init__(self, **kwargs):
        super(Compressor, self, **kwargs).__init__()
        self.name = 'Compressor'
        self.knob_names = ['threshold', 'ratio', 'attackreleaseTime']
        self.knob_ranges = np.array([[-30,0], [1,5], [1e-3,4e-2]])
    def go_wc(self, x, knobs_w):
        return compressor(x, thresh=knobs_w[0], ratio=knobs_w[1], attackrel=knobs_w[2], sr=self.sr), x

class Compressor_4c(Effect):  # compressor with 4 controls
    def __init__(self, **kwargs):
        super(Compressor_4c, self, **kwargs).__init__()
        self.name = 'Compressor_4c'
        self.knob_names = ['threshold', 'ratio', 'attackTime','releaseTime']
        self.knob_ranges = np.array([[-30,0], [1,5], [1e-3,4e-2], [1e-3,4e-2]])
    def go_wc(self, x, knobs_w):
        return compressor_4controls(x, thresh=knobs_w[0], ratio=knobs_w[1], attackTime=knobs_w[2], releaseTime=knobs_w[3], sr=self.sr), x


class Compressor_4c_Large(Effect):  # compressor with 4 controls, larger ranges for parameters
    def __init__(self, **kwargs):
        super(Compressor_4c_Large, self, **kwargs).__init__()
        self.name = 'Compressor_4c_Large'
        self.knob_names = ['threshold', 'ratio', 'attackTime','releaseTime']
        self.knob_ranges = np.array([[-50,0], [1.5,10], [1e-3,1], [1e-3,1]])
    def go_wc(self, x, knobs_w):
        return compressor_4controls(x, thresh=knobs_w[0], ratio=knobs_w[1], attackTime=knobs_w[2], releaseTime=knobs_w[3], sr=self.sr), x


class Comp_Just_Thresh(Effect):  # compressor with just threshold
    """
    Purpose of this effect: used for comparison vs (analog) LA2A
    """
    def __init__(self, **kwargs):
        super(Comp_Just_Thresh, self, **kwargs).__init__()
        self.name = 'Comp_Just_Thresh'
        self.knob_names = ['threshold']
        self.knob_ranges = np.array([[-50,-10]])
        self.ratio = 3.0
        self.attack = .05 # 50ms
        self.release = 1.0 # 1 second!
    def go_wc(self, x, knobs_w):
        return compressor_4controls(x, thresh=knobs_w[0], ratio=self.ratio, attackTime=self.attack, releaseTime=self.release, sr=self.sr), x



class Echo(Effect):
    def __init__(self, **kwargs):
        super(Echo, self, **kwargs).__init__()
        self.name = 'Echo'
        self.knob_names = ['delay_samples', 'ratio', 'echoes']
        #self.knob_ranges = np.array([[100,1500], [0.1,0.9],[2,2]])
        self.knob_ranges = np.array([[400,400], [0.4,1.0],[2,2]])
    def go_wc(self, x, knobs_w):
        return echo(x, delay_samples=int(np.round(knobs_w[0])), ratio=knobs_w[1], echoes=int(np.round(knobs_w[2]))), x

class PitchShifter(Effect):
    def __init__(self, **kwargs):
        super(PitchShifter, self, **kwargs).__init__()
        self.name = 'PitchShifter'
        self.knob_names = ['n_steps']
        self.knob_ranges = np.array([[-12,12]])  # number of 12-tone pitch steps by which to shift the signal
    def go_wc(self, x, knobs_w):
        return librosa.effects.pitch_shift(x, sr=self.sr, n_steps=knobs_w[0]), x   # TODO: librosa's pitch_shift is SLOW!

class Denoise(Effect):  # add noise to x, swap x and y
    """
    This doesn't really denoise: It adds noise to the input, then swaps input & output.
    So you wouldn't be able to input a noisy signal and have it get denoised.
    But when the network trains on this, it learns to take noisy input and denoise it by a tunable amount 'strength'
    """
    def __init__(self, **kwargs):
        super(Denoise, self, **kwargs).__init__()
        self.name = 'Denoise'
        self.knob_names = ['strength']
        self.knob_ranges = np.array([[0.0,0.5]])
        self.is_inverse = True
    def go_wc(self, x, knobs_w):
        return x, x + (knobs_w[0]*(2*np.random.random(x.shape[0])-1)).astype(x.dtype, copy=False)   # swaps y & x: what was the input becomes the output

class DeCompressor_4c(Effect):  # compressor with 4 controls
    def __init__(self):
        super(DeCompressor_4c, self).__init__()
        self.name = 'DeCompressor_4c'
        sub_effect = Compressor_4c()
        self.knob_names = sub_effect.knob_names
        self.knob_ranges = sub_effect.knob_ranges
        self.is_inverse = True          # this effect swaps input & output at the end
    def go_wc(self, x, knobs_w):
        y = compressor_4controls(x, thresh=knobs_w[0], ratio=knobs_w[1], attackTime=knobs_w[2], releaseTime=knobs_w[3])
        return x, y # swap usual order of x and y

class TimeAlign(Effect):  # add noise to x, swap x and y
    """
    This affect completely ignores the input x.  Instead it re-synthesizes a time-aligned y,
    shifts it randomly and outputs that as y
    """
    def __init__(self, sr=44100):
        super(TimeAlign, self).__init__()
        self.name = 'TimeAlign'
        self.knob_names = ['strength']
        self.knob_ranges = np.array([[0.001,0.5]])
        self.is_inverse = True
        chunk_size = 4096 # TODO un-hardcode this
        self.t = np.arange(chunk_size,dtype=np.float32) / sr
    def go_wc(self, x, knobs_w):
        chooser = np.random.choice([2,4,6,7])
        y = synth_input_sample(self.t, chooser, t0_fac=0.5)   # start onset in the middle of chunk
        rand_shift = int(x.shape[0]* knobs_w[0]*(2*np.random.rand()-1)) # shift forward or back by 1/3 of width
        x = np.roll(y,rand_shift)
        if rand_shift > 0:
            x[0:rand_shift] = np.zeros(rand_shift)
        elif rand_shift < 0:
            x[-np.abs(rand_shift):] = np.zeros(np.abs(rand_shift))
        return y, x


class LowPass(Effect):
    # https://gist.github.com/junzis/e06eca03747fc194e322
    def __init__(self, sr=44100):
        super(LowPass, self).__init__()
        self.name = 'LowPass'
        self.knob_names = ['cutoff']
        self.knob_ranges = np.array([[10,2000]])  # number of 12-tone pitch steps by which to shift the signal
        self.sr = sr
    def butter_lowpass(self, cutoff, order=3):
        nyq = 0.5 * self.sr
        normal_cutoff = cutoff / nyq
        b, a = scipy_signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    def go_wc(self, x, knobs_w, order=3):
        b, a = self.butter_lowpass(knobs_w[0], order=order)
        return scipy_signal.lfilter(b, a, x), x


class FileEffect(Effect):
    '''
     'Fake' effect: All this does is grab info ABOUT a dataset of prerecorded files
     corresponding to an effect.  This doesn't actually manipulate any audio, all
     the audio is handled by AudioFileDataSet in data.py

     Relies on a path containing a config file called 'effect_info.ini' and
     Train/ and Val/ subdirectories, as in...

     MyEffectDataPath/
          +-- effect_info.ini
          +-- Train/
          +-- Val/

     Where the format of effect_info.ini is as in this example of an LA2A with a switch:
        [effect]                                             <-- this line has to be exactly like this
        name = LA2A w/ switch                                <--- can have quotes around it or not
        knob_names = ['Limit/Comp', 'Gain', 'Gain Reduction']  <-- list of names, Python format
        knob_ranges = [[0,1], [0,100], [0,100]]               <-- list of lists of min & max knob settings
    '''
    def __init__(self, path, sr=44100, ):
        super(FileEffect, self).__init__()
        print("  FileEffect: path = ",path)
        if (path is None) or (not glob.glob(path+"/Train/target*")) \
            or (not glob.glob(path+"/Val/target*")) or ((not glob.glob(path+"/effect_info.ini"))):
            print(f"Error: can't file target output files or effect_info.ini in path = {path}")
            sys.exit(1)   # Yea, this is fatal

        self.sr = sr
        # read the effect info config file  "effect_info.ini"
        config = configparser.ConfigParser()
        config.read(path+'/effect_info.ini')
        self.name = config['effect']['name']+"(files)"   # tack on "(files)" to the effect name
        #TODO: note that use of 'eval' below could be a potential security issue
        self.knob_names = eval(config.get("effect","knob_names"))
        self.knob_ranges = np.array(eval(config.get("effect","knob_ranges")))
        try:
            self.is_inverse = (True == bool(config['effect']['inverse']) )
            self.name = "De-"+self.name
        except:
            pass   # Ignore errors we don't require that 'inverse' be defined anywhere in the file
    def go_wc(self, x, knobs_w):
        return   # dummy op. there is no plugin to call; we're reading from files

# End of 'effects'

# See data.py for AudioFileDataSet, AudioDataGenerator, etc

# utility routine for effects
def int2knobs(idx:int, knob_ranges:list, settings_per:int) -> list:
  """
  Maps a single (0-indexed) integer to a group of knob settings.
  Useful for systematically covering a range of (linearly) equally-spaced knob
  settings for dataset creation
  NOTE: Operates in a "little-endian" format, i.e. last knob(/digit) varies most
        rapidly as index changes
  Inputs:
     idx:  integer value to convert
     knob_ranges: a list of 2-element lists consiting of [min,max] values for each knob
          Ranges can be anything,
     settings_per: Settings per knob, i.e. number of increments (assumes same inc for all knobs)

  Examples:
  print(int2knobs(12345, [[-0.5,0.5]]*4, 12))
  [0.13636363636363635, -0.40909090909090906, 0.2272727272727273, 0.31818181818181823]

  For rolling a set of 3 dice:
  print( int2knobs(100, [[1,6]]*3, 6))
  [3.0, 5.0, 5.0]

  Simple base 10 aritmetic:
  print(int2knobs(1234, [[0,9]]*4, 10))
  [1.0, 2.0, 3.0, 4.0]
  """
  sp, nk = settings_per, len(knob_ranges)  # mere abbreviations, nk=num_knobs
  assert idx < sp**nk, f"idx ({idx}) must be less than max range of possible values ({sp**nk})"
  knobs = []
  for i in range(nk-1,-1,-1):         # loop over knobs and multiples of sp
    sp_pow = sp**i
    setting = idx // sp_pow        # which setting (of settings_per) for this knob
    ik = nk-1-i                    # because we're going in reverse order, we need to grab knob-ranges in reverse order
    dkval = (knob_ranges[ik][1]-knob_ranges[ik][0])/(sp-1)  # increment of knob value
    knobs.append(knob_ranges[ik][0] + dkval * setting)  # calc knob val and add to list
    idx -= setting * sp_pow        # prepare to calc next "digit" in next loop
  return knobs


if __name__ == "__main__":
    pass
# EOF

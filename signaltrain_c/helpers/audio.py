# -*- coding: utf-8 -*-
__author__ = 'S.H. Hawley'

# imports
import numpy as np
import scipy.signal as signal
import torch
import librosa
from numba import autojit   # Note: nopython version gives symbol errors when used w/ Jupyter Notebook, so using autojit instead
import os

def random_ends(size=1): # probabilty dist. that emphasizes boundaries
    return np.random.beta(0.8,0.8,size=size)

def sliding_window(x, size, overlap=0):
    """
    Stacks 1D array into a series of sliding windows with a certain amount of overlap
    This is fast because it generates a "view" rather than creating a new array.
       overlap = amount of "lookback", when predicting the next set of values
    Example:
        x = np.arange(10)
        print(sliding_window(a, 5, overlap=2))
        [[0 1 2 3 4]
         [3 4 5 6 7]
         [6 7 8 9 0]]
    Source: from last answer to https://stackoverflow.com/questions/4923617/efficient-numpy-2d-array-construction-from-1d-array
    """
    step = size - overlap # in npts
    #nwin = (x.shape[-1]-size)//step + 1  # this truncates any leftover rows, rather than padding with zeros
    nwin = int(np.ceil((x.shape[-1]-size)/step + 1))  # SHH mod to pad with zeros rather than truncate last row
    shape = x.shape[:-1] + (nwin, size)
    strides = x.strides[:-1] + (step*x.strides[-1], x.strides[-1])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

def undo_siding_window(x, overlap):
    """
    Undoes the the sliding window view: Returns 1-D shape of length equal to nearest
        multiple of window, minus overlap
    """
    shape = [(overlap + len(x[:,overlap:])//overlap)*overlap - overlap + 1]
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=[x.strides[-1]], writeable=False)


#--- List of test signals:
def randsine(t, randfunc=np.random.rand, amp_range=[0.5,0.9], freq_range=[5,55], t0_fac=None):
    amp = amp_range[0] + (amp_range[1]-amp_range[0])*randfunc()
    freq = freq_range[0] + (freq_range[1]-freq_range[0])*randfunc()
    t0 = randfunc() * t[-1] if t0_fac is None else t0_fac*t[-1]
    x = amp*np.cos(freq*(t-t0))
    return x

def box(t, randfunc=np.random.rand, t0_fac=None):
    height_low, height_high = 0.3*randfunc()+0.1, 0.35*randfunc() + 0.6
    maxi = len(t)
    delta = 1+ maxi//100     # slope the sides slightly
    i_up = delta+int( 0.3*randfunc() * maxi) if t0_fac is None else int(t0_fac*maxi)
    i_dn = i_up + int( (0.3+0.35*randfunc())*maxi)   # time for jumping back down
    x = height_low*np.ones(t.shape[0]).astype(t.dtype)  # noise of unit amplitude
    x[i_up:i_dn] = height_high
    x[i_up-delta:i_up+delta] = height_low + (height_high-height_low)*(np.arange(2*delta))/2/delta
    x[i_dn-delta:i_dn+delta] = height_high - (height_high-height_low)*(np.arange(2*delta))/2/delta
    return x

def expdecay(t, randfunc=np.random.rand, t0_fac=None):
    t0 = 0.35*randfunc()*t[-1] if t0_fac is None else t0_fac*t[-1]
    height_low, height_high = 0.1*randfunc()+0.1, 0.35*randfunc() + 0.6
    decay = 8*randfunc()
    x = np.exp(-decay * (t-t0)) * height_high   # decaying envelope
    x[np.where(t < t0)] = height_low   # without this, it grow exponentially 'to the left'
    return x

def pluck(t, randfunc=np.random.rand, t0_fac=None):
    x = expdecay(t)
    amp0 = (0.45 * randfunc() + 0.5) * np.random.choice([-1, 1])
    t0 = (2. * randfunc()-1)*0.3 * t[-1] if t0_fac is None else t0_fac*t[-1] # for phase
    freq = 6400. * randfunc()
    return amp0*np.sin(freq * (t-t0)) * x

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
    height = (0.4 * randfunc() + 0.3) * np.random.choice([-1,1])
    width = randfunc()/4 * t[-1]     # half-width actually
    t0 = 2*width + 0.4 * randfunc()*t[-1] if t0_fac is None else t0_fac*t[-1]
    x = height * (1 - np.abs(t-t0)/width)
    x[np.where(t < (t0-width))] = 0
    x[np.where(t > (t0+width))] = 0
    amp_n = (0.1*randfunc()+0.02)   # add noise
    return x + amp_n*(2*np.random.random(t.shape[0])-1)


def read_audio_file(filename, sr=44100):
    signal, sr = librosa.load(filename, sr=sr, mono=True) # convert to mono
    return signal, sr

def readaudio_generator(seconds=2,  path=os.path.expanduser('~')+'/datasets/signaltrain/Val', sr=44100,
    random_every=True):
    """
    reads audio from any number of audio files sitting in directory 'path'
    supplies a window of length "seconds". If random_every=True, this window will be randomly chosen
    """
    # seq_size = amount of audio samples to supply from file
    # basepath = directory containing Train, Val, and Test directories
    # path = audio files for dataset  (can be Train, Val or test)
    # random_every = get a random window every time next is called, or step sequentially through file
    files = os.listdir(path)
    seq_size = seconds * sr
    read_new_file = True
    start = -seq_size
    while True:
        if read_new_file:
            filename = path+'/'+np.random.choice(files)  # pick a random audio file in the directory
            #print("Reading new data from "+filename+" ")
            data, sr = read_audio_file(filename, sr=sr)
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
        chooser = np.random.randint(0, 7)

    if 0 == chooser:                     # sine, with random phase, amp & freq
        return randsine(t, t0_fac=t0_fac)
    elif 1 == chooser:                  # noisy sine
        return randsine(t,t0_fac=t0_fac) + 0.1*(2*np.random.rand(t.shape[0])-1)
    elif 2 == chooser:                    #  "pluck", decaying sine wave
        return pluck(t,t0_fac=t0_fac)
    elif 3 == chooser:                   # ramp up then down
        return triangle(t,t0_fac=t0_fac)
    elif (4 == chooser):                # 'box'
        return box(t,t0_fac=t0_fac)
    elif 5 == chooser:                 # "bunch of spikes"
        return spikes(t)
    elif 6 == chooser:                # noisy box
        return box(t,t0_fac=t0_fac) * (2*np.random.rand(t.shape[0])-1)
    elif 7 == chooser:                # noisy 'pluck'
        amp_n = (0.3*randfunc()+0.1)
        return pluck(t,t0_fac=t0_fac) + amp_n*(2*np.random.random(t.shape[0])-1)  #noise centered around 0
    elif 8 == chooser:                  # just white noise
        amp_n = (0.6*randfunc()+0.2)
        return amp_n*(2*np.random.rand(t.shape[0])-1)
    else:
        return 0.5*(synth_input_sample(t)+synth_input_sample(t)) # superposition of the above
#---- End test signals


#---- Effects

def compressor(x, thresh=-24, ratio=2, attack=2048, dtype=np.float32):
    """
    simple compressor effect, code thanks to Eric Tarr @hackaudio
    Inputs:
       x:        the input waveform
       thresh:   threshold in dB
       ratio:    compression ratio
       attack:   attack & release time (it's a simple compressor!) in samples
    """
    fc = 1.0/float(attack)               # this is like 1/attack time
    b, a = signal.butter(1, fc, analog=False, output='ba')
    zi = signal.lfilter_zi(b, a)

    dB = 20. * np.log10(np.abs(x) + 1e-6).astype(dtype)
    in_env, _ = signal.lfilter(b, a, dB, zi=zi*dB[0])  # input envelope calculation
    out_env = np.copy(in_env)              # output envelope
    i = np.where(in_env >  thresh)          # compress where input env exceeds thresh
    out_env[i] = thresh + (in_env[i]-thresh)/ratio
    gain = np.power(10.0,(out_env-in_env)/20)
    y = (x * gain).astype(dtype)
    return y


@autojit
def compressor_new_fast(x, thresh=-24.0, ratio=2.0, attackTime=0.01,releaseTime=0.01, Fs=44100.0, dtype=np.float32):
    """
    (Minimizing the for loop, removing dummy variables, and invoking numba @autojit made this "fast")
    Inputs:
      x: input signal
      Fs: sample rate in Hz
      thresh: threhold in dB
      ratio: ratio (ratio:1)
      attackTime, releasTime: in seconds
      dtype: typical numpy datatype
    """
    N = len(x)
    y = np.zeros(N, dtype=dtype)
    lin_A = np.zeros(N, dtype=dtype)  # functions as gain

    # Initialize separate attack and release times
    alphaA = np.exp(-np.log(9)/(Fs * attackTime))#.astype(dtype)
    alphaR = np.exp(-np.log(9)/(Fs * releaseTime))#.astype(dtype)

    # Turn the input signal into a uni-polar signal on the dB scale
    x_uni = np.abs(x).astype(dtype)
    x_dB = 20*np.log10(x_uni + 1e-8).astype(dtype)
    # Ensure there are no values of negative infinity
    x_dB = np.clip(x_dB, -96, None)

    # Static Characteristics
    gainChange_dB = np.zeros(x_dB.shape[0])
    i = np.where(x_dB > thresh)
    gainChange_dB[i] =  thresh + (x_dB[i] - thresh)/ratio - x_dB[i] # Perform Downwards Compression

    for n in range(x_dB.shape[0]):   # this loop is slow but unavoidable if alphaA != alphaR. @autojit makes it fast.
        # smooth over the gainChange
        if gainChange_dB[n] < lin_A[n-1]:
            lin_A[n] = ((1-alphaA)*gainChange_dB[n]) +(alphaA*lin_A[n-1]) # attack mode
        else:
            lin_A[n] = ((1-alphaR)*gainChange_dB[n]) +(alphaR*lin_A[n-1]) # release

    lin_A = np.power(10.0,(lin_A/20)).astype(dtype)  # Convert to linear amplitude scalar; i.e. map from dB to amplitude
    y = lin_A * x    # Apply linear amplitude to input sample

    return y.astype(dtype)


 # this is a echo or delay effect
def echo(x, delay_samples=1487, ratio=0.6, echoes=1, dtype=np.float32):
    # ratio = redution ratio
    y = np.copy(x).astype(dtype)
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




# Classes for Effects
class Effect():
    """Generic effect, in this case simple gain
       sub-classed Effects should also define a 'go' method to execute the effect
    """
    def __init__(self, sr=44100):
        self.name = 'Generic Effect'
        self.knob_names = ['knob']
        self.knob_ranges = np.array([[0,1]])  # min,max world coordinate values for "all the way counterclockwise" and "all the way clockwise"
        self.sr = sr

    def knobs_wc(self, knobs_nn):   # convert knob vals from [-.5,.5] to "world coordinates" used by effect functions
        return (self.knob_ranges[:,0] + (knobs_nn+0.5)*(self.knob_ranges[:,1]-self.knob_ranges[:,0])).tolist()

    def info(self):
        assert len(self.knob_names)==len(self.knob_ranges)
        print(f'Effect: {self.name}.  Knobs:')
        for i in range(len(self.knob_names)):
            print(f'                            {self.knob_names[i]}: {self.knob_ranges[i][0]} to {self.knob_ranges[i][1]}')
    # Effects should also define a 'go' method which executes the effect, mapping input and knobs_nn to output y, x
    #   We return x as well as y, because some effects may reverse x & y (e.g. denoiser)


class Compressor(Effect):
    def __init__(self):
        super(Compressor, self).__init__()
        self.name = 'Compressor'
        self.knob_names = ['threshold', 'ratio', 'attackrelease']
        self.knob_ranges = np.array([[-30,0], [1,5], [10,2048]])
    def go(self, x, knobs_nn):
        knobs_w = self.knobs_wc(knobs_nn)
        return compressor(x, thresh=knobs_w[0], ratio=knobs_w[1], attack=knobs_w[2]), x

class Compressor_4c(Effect):  # compressor with 4 controls
    def __init__(self):
        super(Compressor_4c, self).__init__()
        self.name = 'Compressor_4c'
        self.knob_names = ['thresh', 'ratio', 'attackTime','releaseTime']
        self.knob_ranges = np.array([[-30,0], [1,5], [1e-3,4e-2], [1e-3,4e-2]])
    def go(self, x, knobs_nn):
        knobs_w = self.knobs_wc(knobs_nn)
        return compressor_new_fast(x, thresh=knobs_w[0], ratio=knobs_w[1], attackTime=knobs_w[2], releaseTime=knobs_w[3]), x

class Echo(Effect):
    def __init__(self):
        super(Echo, self).__init__()
        self.name = 'Echo'
        self.knob_names = ['delay_samples', 'ratio', 'echoes']
        #self.knob_ranges = np.array([[100,1500], [0.1,0.9],[2,2]])
        self.knob_ranges = np.array([[400,400], [0.4,1.0],[2,2]])
        self.sr = sr
    def go(self, x, knobs_nn):
        knobs_w = self.knobs_wc(knobs_nn)
        return echo(x, delay_samples=int(np.round(knobs_w[0])), ratio=knobs_w[1], echoes=int(np.round(knobs_w[2]))), x

class PitchShifter(Effect):
    def __init__(self):
        super(PitchShifter, self).__init__()
        self.name = 'Pitch Shifter'
        self.knob_names = ['n_steps']
        self.knob_ranges = np.array([[-12,12]])  # number of 12-tone pitch steps by which to shift the signal
    def go(self, x, knobs_nn):
        knobs_w = self.knobs_wc(knobs_nn)
        return librosa.effects.pitch_shift(x, sr=self.sr, n_steps=knobs_w[0]), x   # TODO: librosa's pitch_shift is SLOW!

class Denoise(Effect):  # add noise to x, swap x and y
    """
    This doesn't really denoise: It adds noise to the input, then swaps input & output.
    So you wouldn't be able to input a noisy signal and have it get denoised.
    But when the network trains on this, it learns to take noisy input and denoise it by a tunable amount 'strength'
    """
    def __init__(self):
        super(Denoise, self).__init__()
        self.name = 'Denoise'
        self.knob_names = ['strength']
        self.knob_ranges = np.array([[0.01,0.5]])
    def go(self, x, knobs_nn):
        knobs_w = self.knobs_wc(knobs_nn)
        return x, x + knobs_w[0]*(2*np.random.random(x.shape[0])-1)   # swaps y & x: what was the input becomes the output

# End of effects


# Data Generator
class AudioDataGenerator():
    def __init__(self, time_series_length, sampling_freq, effect, batch_size=10, requires_grad=True, device=torch.device("cuda:0")):
        super(AudioDataGenerator, self).__init__()
        self.time_series_length = time_series_length
        self.t = np.arange(time_series_length,dtype=np.float32) / sampling_freq
        self.effect = effect
        self.batch_size = batch_size
        self.requires_grad = requires_grad
        self.device = device

        # preallocate memory
        self.x = np.zeros((batch_size,time_series_length),dtype=np.float32)
        self.y = np.zeros((batch_size,time_series_length),dtype=np.float32)
        self.knobs = np.zeros((batch_size,len(self.effect.knob_ranges)),dtype=np.float32)

    def gen_single(self, chooser=None, knobs=None, recyc_x=None):
        """create a single time-series"""
        if chooser is None:
            chooser = np.random.choice([0,1,2,4,6,7])  # for compressor
            #chooser = np.random.choice([1,3,5,6,7])  # for echo

        if recyc_x is None:
            x = synth_input_sample(self.t, chooser)
        else:
            x = recyc_x   # don't generate new x

        if knobs is None:
            knobs = random_ends(len(self.effect.knob_ranges))-0.5  # inputs to NN, zero-mean...except we emphasize the ends slightly

        y, x = self.effect.go(x, knobs)

        return x, y, knobs

    def new(self,chooser=None, knobs=None, recyc_x=False):  # Generate new x, y, knobs  set
        # was going to try parallel via multiprocessing but it's actually slower than serial
        #self.pool = Pool(processes=10)
        knobs = None #random_ends(len(self.effect.knob_ranges))-0.5  # same knobs for whole batch
        for line in range(self.batch_size):
            if recyc_x:
                #self.x[line,:], self.y[line,:], self.knobs[line,:] = self.pool.apply_async(partial(self.gen_single, chooser, knobs, recyc_x=self.x[line,:])).get()
                self.x[line,:], self.y[line,:], self.knobs[line,:] = self.gen_single(chooser, knobs=knobs, recyc_x=self.x[line,:])
            else:
                #self.x[line,:], self.y[line,:], self.knobs[line,:] = self.pool.apply_async(partial(self.gen_single,chooser,knobs)).get()
                self.x[line,:], self.y[line,:], self.knobs[line,:] = self.gen_single(chooser, knobs=knobs)
        #pool.close()

        # Turn numpy data into torch/cuda data
        x_torch = torch.autograd.Variable(torch.from_numpy(self.x).to(self.device), requires_grad=self.requires_grad).float()
        y_torch = torch.autograd.Variable(torch.from_numpy(self.y).to(self.device), requires_grad=False).float()
        knobs_torch =  torch.autograd.Variable(torch.from_numpy(self.knobs).to(self.device), requires_grad=self.requires_grad).float()
        return x_torch, y_torch, knobs_torch

    def new_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.x = np.zeros((batch_size,self.time_series_length),dtype=np.float32)
        self.y = np.zeros((batch_size,self.time_series_length),dtype=np.float32)
        self.knobs = np.zeros((batch_size,len(self.effect.knob_ranges)),dtype=np.float32)

if __name__ == "__main__":
    pass
# EOF

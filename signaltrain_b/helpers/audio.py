# -*- coding: utf-8 -*-
__author__ = 'S.H. Hawley'

# imports
import numpy as np
import scipy.signal as signal


def synth_input_sample(t, chooser=None):
    """
    Synthesizes various 'fake' audio wave forms -- synthetic data
    """
    x = np.copy(t)*0.0
    if chooser is None:
        chooser = np.random.randint(0, 6)

    if 0 == chooser:
        amp = 0.4+0.4*np.random.random()
        freq = 30*np.random.random()    # sin, with random start & freq
        t0 = np.random.random()
        x = amp*np.sin(freq*(t-t0))
        x[np.where(t < t0)] = 0.0
        return x

    elif 1 == chooser:                # fixed sine wave
        freq = 5+150*np.random.random()
        amp = 0.4+0.4*np.random.random()
        global global_freq
        global_freq = freq
        return amp*np.sin(freq*t)

    elif 2 == chooser:                    #  "pluck"
        amp0 = (0.6 * np.random.random() + 0.3) * np.random.choice([-1, 1])
        t0 = (2. * np.random.random()-1)*0.3
        freq = 6400. * np.random.random()
        x = amp0*np.sin(freq * (t-t0))
        decay = 8*np.random.random()
        x = np.exp(-decay * (t-t0)) * x   # decaying envelope
        x[np.where(t < t0)] = 0           # without this, it grow exponentially 'to the left'
        return x

    elif 3 == chooser:                   # ramp up then down
        height = (0.4 * np.random.random() + 0.2) * np.random.choice([-1,1])
        width = 0.3 * np.random.random()/4      # half-width actually
        t0 = 2*width + 0.4 * np.random.random() # make sure it fits
        x = height * (1 - np.abs(t-t0)/width)
        x[np.where(t < (t0-width))] = 0
        x[np.where(t > (t0+width))] = 0
        return x

    elif 4 == chooser:                  # 'box'
        height = (0.3*np.random.random()+0.2)*np.random.choice([-1,1])
        x = height*np.ones(t.shape[0])
        t1 = np.random.random()/2
        t2 = t1 + np.random.random()/2
        x[np.where(t < t1)] = 0.0
        x[np.where(t > t2)] = 0.0
        return x

    elif 5 == chooser:                 # "bunch of spikes"
        n_spikes = 100
        for i in range(n_spikes):   # arbitrarily make a 'spike' somewhere, surrounded by silence
          loc = int(np.random.random()*len(t)-2)+1
          height = np.random.random()-0.5    # -0.5...0.5
          x[loc] = height
          x[loc+1] = height/2  # widen the spike a bit
          x[loc-1] = height/2
        x = x + 0.1*np.random.normal(0.0, scale=0.1, size=x.size)    # throw in noise
        return x

    elif 6 == chooser:                # white noise
        amp = 0.2+0.2*np.random.random()
        x = amp*(2*np.random.random(t.shape[0])-1)
        return x

    elif 7 == chooser:                # noisy 'pluck'
        amp0 = (0.6*np.random.random()+0.4)*np.random.choice([-1,1])
        t0 = np.random.random()*0.5   # start late by some amount
        freq = 6400*np.random.random()
        x = amp0*np.sin(freq* (t-t0))
        amp_n = (0.4*np.random.random()+0.2)
        noise = amp_n*(2*np.random.random(t.shape[0])-1)  #noise centered around 0
        x += noise
        decay = 8*np.random.random()
        x = np.exp(-decay * (t-t0)) * x   # decaying envelope
        x[np.where(t < t0)] = 0   # without this, it grow exponentially 'to the left'
        return x

    else:
        x = 0.5*(synth_input_sample(t)+synth_input_sample(t)) # superposition of previous
        return x


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
    out_env = np.copy(in_env).astype(dtype)               # output envelope
    i = np.where(in_env >  thresh)          # compress where input env exceeds thresh
    out_env[i] = thresh + (in_env[i]-thresh)/ratio
    gain = np.power(10.0,(out_env-in_env)/10).astype(dtype)
    y = (np.copy(x) * gain).astype(dtype)
    return y

# EOF

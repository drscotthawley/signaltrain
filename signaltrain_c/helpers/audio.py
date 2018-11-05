
# -*- coding: utf-8 -*-
__author__ = 'S.H. Hawley'

# imports
import numpy as np
import scipy.signal as signal

def random_ends(size=1): # probabilty dist. that emphasizes boundaries
    return np.random.beta(0.85,0.85,size=size)

def synth_input_sample(t, chooser=None, randfunc=random_ends):
    """
    Synthesizes various 'fake' audio wave forms -- synthetic data
    """
    x = np.copy(t)*0.0
    if chooser is None:
        chooser = np.random.randint(0, 7)

    if 0 == chooser:                     # sin, with random phase, amp & freq
        amp = 0.4+0.6*randfunc()
        freq = 40+150*randfunc()
        t0 = randfunc() * t[-1]
        x = amp*np.cos(freq*(t-t0))
        return x

    elif 1 == chooser:                # fixed sine wave
        freq = 5+150*randfunc()
        amp = 0.4+0.6*randfunc()
        return amp*np.sin(freq*t)

    elif 2 == chooser:                    #  "pluck"
        amp0 = (0.6 * randfunc() + 0.4) * np.random.choice([-1, 1])
        t0 = (2. * randfunc()-1)*0.3 * t[-1]
        freq = 6400. * randfunc()
        x = amp0*np.sin(freq * (t-t0))
        decay = 8*randfunc()
        x = np.exp(-decay * (t-t0)) * x   # decaying envelope
        x[np.where(t < t0)] = 0           # without this, it grow exponentially 'to the left'
        return x

    elif 3 == chooser:                   # ramp up then down
        height = (0.4 * randfunc() + 0.2) * np.random.choice([-1,1])
        width = 0.3 * randfunc()/4 * t[-1]     # half-width actually
        t0 = 2*width + 0.4 * randfunc() # make sure it fits
        x = height * (1 - np.abs(t-t0)/width)
        x[np.where(t < (t0-width))] = 0
        x[np.where(t > (t0+width))] = 0
        return x

    elif (4 == chooser):                # 'noisy box'
        height_low = 0.3*randfunc()+0.1
        height_high = 0.7 + 0.3*randfunc()
        t_up = randfunc()/3 * t[-1]               # time where volume jumps up
        t_dn = t_up + 0.7*randfunc()* t[-1]    # time for jumping back down
        t_mid = 0.5*(t_up+t_dn)
        #x = 2*np.random.random(t.shape[0]).astype(t.dtype)-1  # noise of unit amplitude
        x = 2*np.ones(t.shape[0]).astype(t.dtype)-1  # noise of unit amplitude

        x[np.where(abs(t-t_mid) <= (t_dn-t_mid))] *= height_high # inside the box
        x[np.where(abs(t-t_mid) >  (t_dn-t_mid))] *= height_low  # outside the box
        return x

    elif 5 == chooser:                 # "bunch of spikes"
        n_spikes = 100
        for i in range(n_spikes):   # arbitrarily make a 'spike' somewhere, surrounded by silence
          loc = int( int(randfunc()*len(t)-2)+1* t[-1] )
          height = randfunc()-0.5    # -0.5...0.5
          x[loc] = height
          x[loc+1] = height/2  # widen the spike a bit
          x[loc-1] = height/2
        x = x + 0.1*np.random.normal(0.0, scale=0.1, size=x.size)    # throw in noise
        return x

    elif 6 == chooser:                # white noise
        amp = 0.2+0.2*randfunc()
        x = amp*(2*np.random.random(t.shape[0])-1)
        return x

    elif 7 == chooser:                # noisy 'pluck'
        amp0 = (0.6*randfunc()+0.4)*np.random.choice([-1,1])
        t0 = (randfunc()*0.4)*t[-1]   # start late by some amount
        freq = 640*randfunc()
        x = amp0*np.sin(freq* (t-t0))
        amp_n = (0.4*randfunc()+0.2)
        noise = amp_n*(2*np.random.random(t.shape[0])-1)  #noise centered around 0
        x += noise
        decay = 8*randfunc()
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


# code from Eric Tarr
def compressor_new(x,Fs=44100,thresh=-24.0, ratio=2.0, attackTime=0.01,releaseTime=0.01, dtype=np.float32):
    """
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
    lin_A = np.zeros(N, dtype=dtype)

    # Initialize separate attack and release times
    alphaA = np.exp(-np.log(9)/(Fs * attackTime)).astype(dtype)
    alphaR = np.exp(-np.log(9)/(Fs * releaseTime)).astype(dtype)

    gainSmoothPrev = 0 # Initialize smoothing variable

    # Loop over each sample to see if it is above thresh
    for n in np.arange(N):
        # Turn the input signal into a uni-polar signal on the dB scale
        x_uni = np.abs(x[n]).astype(dtype)
        x_dB = 20*np.log10(x_uni + 1e-8).astype(dtype)
        # Ensure there are no values of negative infinity
        if x_dB < -96:
            x_dB = -96

        # Static Characteristics
        if x_dB > thresh:
            gainSC = thresh + (x_dB - thresh)/ratio # Perform Downwards Compression
        else:
            gainSC = x_dB # Do not perform compression

        gainChange_dB = gainSC - x_dB

        # smooth over the gainChange
        if gainChange_dB < gainSmoothPrev:
            # attack mode
            gainSmooth = ((1-alphaA)*gainChange_dB) +(alphaA*gainSmoothPrev)
        else:
            # release
            gainSmooth = ((1-alphaR)*gainChange_dB) +(alphaR*gainSmoothPrev)

        # Convert to linear amplitude scalar
        lin_A[n] = np.power(10.0,(gainSmooth/20)).astype(dtype)

        # Apply linear amplitude to input sample
        y[n] = lin_A[n] * x[n]

        # Update gainSmoothPrev used in the next sample of the loop
        gainSmoothPrev = gainSmooth

    return y.astype(dtype)

# EOF

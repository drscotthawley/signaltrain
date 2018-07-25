#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Scott H. Hawley'
__copyright__ = 'Scott H. Hawley'

"""
Draft. Just a demo for now.

Visualization of model activations (and weights)
-  in realtime, based on live microphone input - 'oscilloscope-like' display
-  TODO: from audio file(s)

Still under development.  Currently it just reads from the (live) default microphone,
multiplies by one pre-fabriated array of weights and shows the result.
Later I will try to load layers from a model, e.g. from checkpoint files

Tested only on Mac OS X High Sierra with Python 3.6 (Anaconda)
"""

import numpy as np
import torch
import argparse
import signaltrain as st
import cv2                  # pip install opencv-python
import soundcard as sc      # pip install git+https://github.com/bastibe/SoundCard
from scipy.ndimage.interpolation import shift  # used for oscilloscope trigger


# Gobal parameters...   TODO: move these elswhere...

# Define the layers & their sizes
# Layer sizes - lengths of the activation (output) of each layer
# This is assuming a 'Sequential' model.  No skip connections.
#     - Input counts as layer 0
#     - Layer 1 maps
#  TODO: add option of activation functions (RelU, etc). So far all are linear activations
#  TODO: get rid of all global variables!!
layer_act_dims = [1024,1024,1024]#,512]
n_weights = len(layer_act_dims)-1
print("n_weights = ",n_weights)
weights_dims =[]
for i in range(n_weights):
    weights_dims.append( [layer_act_dims[i],layer_act_dims[i+1]])
print("weights_dims =",weights_dims)


imWidth = max(layer_act_dims)    # image width for 'oscilloscope display'
imHeight = 600                   # image height for oscilloscope display; can be anything.

layer_in_dim, layer_out_dim = layer_act_dims[0], layer_act_dims[1] # TODO: this is legacy from earlier version

# OpenCV BGR colors
blue, green, cyan = (255,0,0), (0,255,0), (255,255,0)
#OpenCV colormaps
blackwhite, rainbow, heat = cv2.COLORMAP_BONE, cv2.COLORMAP_JET, cv2.COLORMAP_AUTUMN



"""
Routines that draw weights
"""
# draw weights for one layer
def draw_weights_layer(weights_layer, title="weights_layer", colormap=rainbow):
    img = np.clip(weights_layer*255 ,-255,255).astype(np.uint8)    # scale
    img = np.repeat(img[:,:,np.newaxis],3,axis=2)            # add color channels
    img = cv2.applyColorMap(img, colormap)           # rainbow, blue=low, red=high
    window = cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)                      # show what we've got
    cv2.resizeWindow(title, int(imWidth/2),int(imWidth/2))   # zoom out (can use two-finger-scroll to zoom in)

# draw weights for all layers
def draw_weights(weights, title="weights", colormap=rainbow):
    for l in range(len(weights)):
        draw_weights_layer(weights[l], title="weights_layer"+str(l))



def random_color(seed=None):
    # seed is a cheap way of keeping the color from changing too frequently
    if seed is not None:
        np.random.seed(seed+1)     # +1 because it gives you green for seed=0   :-)
    return (128+np.random.randint(127), 128+np.random.randint(127), 128+np.random.randint(127) )


"""
This actually draws the 'oscilloscope' display for the various activations
"""
def draw_activations(screen, weights, mono_audio, xs, \
    title="activations (cyan=input, green=output)", gains=[3,0.3]):
    screen *= 0                                # clear the screen
    mono_audio *= gains[0]                     # apply gain before activation

    n_weights = len(weights)
    max_amp_pixels = imHeight/(n_weights+1)/2    # maximum amplitude in pixels
    dy0 = 2*max_amp_pixels                     # spacing between zero lines

    # Input layer
    act = mono_audio                             # first activation is just the input
    y0 = max_amp_pixels                          # zero line
    # minux sign in the following is because computer graphics are 'upside down'
    ys_in = ( y0 - max_amp_pixels * np.clip( act[0:len(xs)], -1, 1) ).astype(np.int)
    pts_in = np.array(list(zip(xs,ys_in)))      # pair up xs & ys for input
    cv2.polylines(screen,[pts_in],False,cyan)   # draw lines connecting the points

    # now compute and show other layer activations
    for l in range(n_weights):
        y0 = max_amp_pixels + (l+1)*dy0                         # zero line
        act = np.matmul(act, weights[l])        # activations are a new waveform
        ys_out = ( y0 - max_amp_pixels * np.clip(  act[0:len(xs)], -1, 1) ).astype(np.int) # gains[1] gets applied to weights directly
        ys_out = ys_out[0:layer_out_dim]            # don't show more than is supposed to be there
        pts_out = np.array(list(zip(xs,ys_out)))    # pair up xs & ys for output
        cv2.polylines(screen,[pts_out],False, random_color(l))

    # draw window showing activations
    window = cv2.namedWindow(title,cv2.WINDOW_NORMAL)   # allow the window containing the image to be resized
    cv2.imshow(title, screen.astype(np.uint8))
    return


"""
this is a trigger function for the oscilloscope
"""
def find_trigger(mono_audio, thresh=0.02, pos_slope=True):  # thresh = trigger level
    start_ind = None     # this is where in the buffer the trigger should go; None
    shift_forward = shift(mono_audio, 1, cval=0)
    if pos_slope:
        inds = np.where(np.logical_and(mono_audio >= thresh, shift_forward <= thresh))
    else:
        inds = np.where(np.logical_and(mono_audio <= thresh, shift_forward >= thresh))
    if (len(inds[0]) > 0):
        start_ind = inds[0][0]
    return start_ind


"""
Just prints the keys one can use. wish I could get arrow keys working
"""
def instructions():
    print("Keys: ")
    print("  = : increase input gain")
    print("  ' : decrease input gain")
    print("  ] : increase output gain")
    print("  [ : decrease output gain")
    print("  - : increase trigger level")
    print("  p : decrease trigger level")
    print("      Two-finger scroll will zoom in")
    print("")
    print("Note: windows start out reduced in display size; can be resized at will.")
    #print("      (Don't beleive the 'Zoom:%' display; it doesn't reflect proper array size)")


"""
# acquires (syntesizes or reads in) all weights of networks
"""
def get_weights(names, layer_dims):
    # name: a string selector for the type of weights initialization
    # in_layer_dim, out_layer_dim: dimensions of the weights array
    assert len(names) == len(layer_dims)-1,"names needs to be one less than layer_dims"

    weights = []
    print("layer_dims =",layer_dims)
    n_weights = len(layer_dims)-1
    print("n_weights = ",n_weights)
    for l in range(n_weights):
        name = names[l]
        print("     get_weights: l=",l)
        in_layer_dim, out_layer_dim = layer_dims[l], layer_dims[l+1]
        ny, nx = (in_layer_dim, out_layer_dim)

        if ('cos' == name):
            x = np.linspace(0, 1, nx)
            y = np.linspace(0, 1, ny)
            xv, yv = np.meshgrid(x,y)
            weights.append(0.5*np.cos(2*3.14*4*xv)*np.cos(2*3.14*4*yv))
            #weights += 0.1*np.random.rand(layer_in_dim, layer_out_dim)
        elif ('rand' == name):
            weights.append(np.random.rand(ny, nx)-0.5)
        elif ('FNNAnalysis' == name):
            freq_subbands = 256
            hop_size = 1024
            fs = 44100.
            sig_length = 8192
            expected_time_frames = sig_length/float(hop_size) + 1

            net = st.transforms.FNNAnalysis(ft_size=freq_subbands)
            w = net.fnn_analysis_real.weight.data.numpy()
            print("weight data = ")
            for param in net.parameters():
                pass
                 #print("FNNAnalysis param = ",param)
            w = param.data.numpy()
            #weights.append(st.transforms.core_modulation(ny, nx))
            weights.append(w)
            signal = torch.from_numpy(np.random.rand(8192).astype(np.float32))
            out = net(signal)
            print("out.size = ",out.size())
        elif ('FNNSynthesis' == name):
             weights.append(np.transpose(st.transforms.core_modulation(ny, nx)))
        elif ('ft' == name):
            w = np.fft.fft(np.eye(in_layer_dim))
            w = np.real(w)
            weights.append(w)
        elif ('ift' == name):
            weights.append(np.fft.ifft(np.eye(in_layer_dim)))

    return weights


"""
# 'Oscilloscope' routine; audio buffer & sample rate; make the audio buffer a little bigger than 'needed',
#  to avoid showing zero-pads (which are primarily there for 'safety')
"""
def scope(weights, buf_size=2000, fs=44100):

    default_mic = sc.default_microphone()
    print("oscilloscope: listening on ",default_mic)
    instructions()

    trig_level = 0.001   # trigger value for input waveform
    gains = [10,1]    # gains for input and output

    # allocate storage for 'screen'
    screen = np.zeros((imHeight,imWidth,3), dtype=np.uint8) # 3=color channels
    xs = np.arange(imWidth).astype(np.int)                  # x values of pixels (time samples)
    draw_weights(weights)

    while (1):                             # keep looping until someone stops this
        with default_mic.recorder(samplerate=fs) as mic:
            audio_data = mic.record(numframes=buf_size)  # get some audio from the mic

        bgn = find_trigger(audio_data[:,0], thresh=trig_level)    # try to trigger
        layer_in_dim = layer_act_dims[0]                     # length of input layer
        if bgn is not None:
            end = min(bgn+layer_in_dim, buf_size)                 # don't go off the end of the buffer
            pad_len = max(0, layer_in_dim - (end-bgn) )           # might have to pad with zeros
            padded_data = np.pad(audio_data[bgn:end,0],(0,pad_len),'constant',constant_values=0)
            draw_activations(screen, weights, padded_data, xs, gains=gains)             # draw left channel
        else:
            draw_activations(screen, weights, audio_data[0:layer_in_dim,0]*0, xs, gains=gains)   # draw zero line


        key = cv2.waitKeyEx(1) & 0xFF         # keyboard input

        # Couldn't get arrow keys to work.
        if (key != -1) and (key !=255):
            print('key = ',key)
        if ord('q') == key:       # quit key
            break
        elif ord('=') == key:  # equal sign
            gains[0] *= 1.1
            print("gains =",gains)
        elif ord("'") == key:  # signle quote
            gains[0] *= 0.9
            print("gains =",gains)
        elif ord(']') == key:  #  right bracket
            gains[1] *= 1.1
            print("gains =",gains)
            weights = [ x*1.1 for x in weights ]
            draw_weights(weights)
        elif ord('[') == key:  # left bracket
            gains[1] *= 0.9
            weights = [ x*0.9 for x in weights ]
            print("gains =",gains)
            draw_weights(weights)
        elif ord('-') == key:     # minus sign
            trig_level += 0.001
            print("trig_level =",trig_level)
        elif ord("p") == key:     # letter p
            trig_level -= 0.001
            print("trig_level =",trig_level)
    return


def main():
    """
    Set up the 't ransform' weights  TODO: load these from PyTorch model
    """
    weights = get_weights(['cos','rand'], layer_act_dims)    # true fourier transform
    #weights = get_weights(['FNNAnalysis','FNNSynthesis'], layer_act_dims)   # stylianos' layers


    """
    Call the oscilloscope in order to visualize activations
    """
    scope(weights, buf_size=int(layer_act_dims[0]*1.5))

    """
    # Note: by using multiprocessing or threading, could run scope & other visualizations simultaneously
    # not sure how the audio library would like that maybe via threading so they all share the same
    # input buffer
    """

    return


if __name__ == '__main__':
    main()

# EOF

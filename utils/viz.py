#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Scott H. Hawley'
__copyright__ = 'Scott H. Hawley'

"""

Visualization of model activations (and weights)
-  in realtime, based on live microphone input - 'oscilloscope-like' display
-  TODO: from audio file(s)

Still under development.  Currently it just reads from the (live) default microphone,
multiplies by one pre-fabriated array of weights and shows the result.
Later I will try to load layers from a model, e.g. from checkpoint files

Example usage:
    ./viz.py ../demo/modelcheckpoint_4c.tar

TODO/Limitations:
   - Assumes four knobs.
   - Allow user to change knobs.  (Defaults to middle of knob range)

Tested only on Mac OS X High Sierra with Python 3.6/3.7 (Anaconda)
"""

import numpy as np
import os, sys
import torch
import torch.nn as nn
import argparse
import cv2                  # pip install opencv-python
import soundcard as sc      # pip install git+https://github.com/bastibe/SoundCard
from scipy.ndimage.interpolation import shift  # used for oscilloscope trigger

# Code for model
sys.path.append('../signaltrain')
import nn_proc
TheModelClass = nn_proc.st_model
from ptsd2full import load_model  # ptsd2full is in utils directory with viz

# Where to do PyTorch Calcs: CPU or GPU?
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')



# Global variables for graphics windows
imWidth = 1024    # image width for 'oscilloscope display'
imHeight = 600                   # image height for oscilloscope display; can be anything.
blue, green, cyan, yellow = (255,0,0), (0,255,0), (255,255,0), (0,255,255)   # OpenCV BGR colors
blackwhite, rainbow, heat = cv2.COLORMAP_BONE, cv2.COLORMAP_JET, cv2.COLORMAP_AUTUMN  #OpenCV colormaps

layer_act_dims = [1024]

# a few other globals that are kind of handy
knob_names, knob_ranges, knobs_nn = [], [], []
knob_controls_window = 'effect knob controls'
cv2.namedWindow(knob_controls_window)
logo = cv2.imread('stlogo.png',0)  # if you don't give it a picture you get a black rectangle
logo = np.pad(logo, ((25,0),(150,150)), mode='edge')
cv2.imshow(knob_controls_window, logo)


def check_window_exists(title):
    try:
        window_handle = cv2.getWindowProperty(title, 0)
        #print(f"window {title}: handle is {window_handle}")
        return window_handle == 0.0
        return True
    except:
        print("******  error checking for window ",title)
        return False


"""
Routines that draw 2d functions as images
"""
# draw weights for one layer, using one window
def show_2d(array_2d, title="weights_layer", colormap=rainbow, flip=True):

    #print("weights_layer.shape = ",weights_layer.shape)
    if len(array_2d.shape) < 2:
        return
    img = np.clip(array_2d*255 ,-255,255).astype(np.uint8)   # scale
    if flip:
        img = np.flip(np.transpose(img))
    img = np.repeat(img[:,:,np.newaxis],3,axis=2)            # add color channels
    img = cv2.applyColorMap(img, colormap)                   # rainbow: blue=low, red=high
    # see if it exists
    new_window = not check_window_exists(title)
    window = cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    if new_window:
        cv2.resizeWindow(title, img.shape[1], img.shape[0])                      # show what we've got
    #aspect = img.shape[0] / img.shape[1]
    #if aspect > 3:
    #    cv2.resizeWindow(title, 200, 1024)   # zoom in/out (can use two-finger-scroll to zoom in)
        #print(f"size for {title} = 1024, {int(1024/aspect*img.shape[0])}")
    #else:
    #    cv2.resizeWindow(title, int(imWidth/2),int(imWidth/2))   # zoom in/out (can use two-finger-scroll to zoom in)


# draw weights for all layers using model state dict
def draw_weights(model, colormap=rainbow):
    sd = model.state_dict()
    for key, value in sd.items():
        show_2d(value.data.squeeze().numpy(), title=key+" "+str(value.data.numpy().shape))


def random_color(seed=None):
    # seed is a cheap way of keeping the color from changing too frequently
    if seed is not None:
        np.random.seed(seed+1)     # +1 because it gives you green for seed=0   :-)
    return (128+np.random.randint(127), 128+np.random.randint(127), 128+np.random.randint(127) )


"""
This actually draws the 'oscilloscope' display for the various 1d activations
but also draws 2d activations

...NOTE: for now, we only show input and output
"""
def draw_activations(screen, model, mono_audio, xs, trig_level=None, \
        title="activations (cyan=input, green=output)", gains=[1.0,1.0]):

    # run the model
    x = torch.as_tensor(mono_audio).unsqueeze(0).double().requires_grad_(False)
    knobs = torch.as_tensor(knobs_nn).unsqueeze(0).double().requires_grad_(False)
    #torch.zeros(model.num_knobs, requires_grad=False).unsqueeze(0).double()  # TODO: allow user to change knobs

    y_hat, mag, mag_hat, layer_acts = model.forward(x, knobs, return_acts=True)



    screen *= 0                                # clear the screen

    # parameters for the one window that will show all the audio activations together
    # count how many activations are 1d
    n_1d_acts = 0                               # number of activations to show in oscilloscope besides the input
    for l in range(len(layer_acts) ):
        if len(layer_acts[l].data.squeeze().detach().numpy().shape)==1:
            n_1d_acts += 1
    max_amp_pixels = imHeight/(n_1d_acts+1)/2    # maximum amplitude in pixels
    dy0 = 2*max_amp_pixels                     # spacing between zero lines

    # Draw Input audio
    act = mono_audio                             # first show the input
    y0 = max_amp_pixels                          # zero line
    # minux sign after y0 in the following is because computer graphics are 'upside down'
    ys_in = ( y0 - max_amp_pixels * np.clip( act[-len(xs):], -1, 1) ).astype(np.int)
    pts_in = np.array(list(zip(xs,ys_in)))      # pair up xs & ys for input
    cv2.polylines(screen,[pts_in],False,cyan)   # draw lines connecting the points

    if trig_level is not None:  # draw the trigger
        trig_pos = int(y0 - trig_level*max_amp_pixels)
        pts_in = np.array(list(zip([0,10],[trig_pos,trig_pos] ) ) )
        cv2.polylines(screen,[pts_in],False,yellow)


    # draw all other activations (besides input)
    n_acts = len(layer_acts)
    count_1d = 0
    for l in range(len(layer_acts) ):
        act = layer_acts[l].data.squeeze().detach().numpy()
        #print(f"layer {l} of {n_acts}: act.shape = {act.shape}, len = {len(act.shape)}")
        if len(act.shape)==1:
            count_1d += 1
            y0 = max_amp_pixels + (count_1d)*dy0                         # zero line
            act *= gains[1]
            ys_out = ( y0 - max_amp_pixels * np.clip(  act[-len(xs):], -1, 1) ).astype(np.int) # gains[1] gets applied to weights directly
            #ys_out = ys_out[0:layer_out_dim]            # don't show more than is supposed to be there
            pts_out = np.array(list(zip(xs,ys_out)))    # pair up xs & ys for output
            if (count_1d < n_1d_acts):
                color = random_color()
            else:
                color = green
            cv2.polylines(screen,[pts_out],False, color)
        if len(act.shape) == 2:
            show_2d(act, title=f"act_{l} {act.shape}")

    #cv2.text(screen, 'gains = '+str(gains))
    cv2.putText(screen,f"gains = {gains[0]:.1f}, {gains[1]:.1f} ", (10,30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), lineType=cv2.LINE_AA)


    # draw the one window showing all the activations together
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
    print("  Q : quit ")
    print("  = : increase input gain")
    print("  - : decrease input gain")
    print("  ] : increase output gain")
    print("  [ : decrease output gain")
    print("  ' : increase trigger level")
    print("  ; : decrease trigger level")
    print("      Two-finger scroll on trackpad will zoom in on 2D images")
    print("")
    print("Note: windows start out reduced in display size; can be resized at will.")
    #print("      (Don't beleive the 'Zoom:%' display; it doesn't reflect proper array size)")


"""
# 'Oscilloscope' routine; audio buffer & sample rate; make the audio buffer a little bigger than 'needed',
#  to avoid showing zero-pads (which are primarily there for 'safety')
Triggers on input audio
"""
def scope(model, buf_size=2000, fs=44100):

    default_mic = sc.default_microphone()
    print("oscilloscope: listening on ",default_mic)
    instructions()

    trig_level = 0.01   # trigger value for input waveform
    gains = [1,1]    # gains for input and output

    # allocate storage for 'screen'
    screen = np.zeros((imHeight,imWidth,3), dtype=np.uint8) # 3=color channels
    xs = np.arange(imWidth).astype(np.int)                  # x values of pixels (time samples)

    while (1):                             # keep looping until someone stops this
        try:  # sometimes the mic will give a RunTimeError while you're reizing windows
            with default_mic.recorder(samplerate=fs) as mic:
                audio_data = mic.record(numframes=buf_size)  # get some audio from the mic
            audio_data *= gains[0]                     # apply gain before activation

            bgn = find_trigger(audio_data[:,0], thresh=trig_level)    # try to trigger
            layer_in_dim = model.in_chunk_size                     # length of input layer
            if bgn is not None:                                  # we found a spot to trigger at
                end = min(bgn+layer_in_dim, buf_size)                 # don't go off the end of the buffer
                pad_len = max(0, layer_in_dim - (end-bgn) )           # might have to pad with zeros
                padded_data = np.pad(audio_data[bgn:end,0],(0,pad_len),'constant',constant_values=0)
                draw_activations(screen, model, padded_data, xs, trig_level=trig_level, gains=gains)             # draw left channel
            else:
                draw_activations(screen, model,  audio_data[0:layer_in_dim,0]*0, xs, trig_level=trig_level, gains=gains)  # just draw zero line


            key = cv2.waitKeyEx(1) & 0xFF         # keyboard input

            # Controls:  (Couldn't get arrow keys to work.)
            if (key != -1) and (key !=255):
                #print('key = ',key)
                pass
            if ord('q') == key:       # quit key
                break
            elif ord('=') == key:
                gains[0] *= 1.1
            elif ord("-") == key:
                gains[0] *= 0.9
            elif ord(']') == key:  #  right bracket
                gains[1] *= 1.1
            elif ord('[') == key:  # left bracket
                gains[1] *= 0.9
            elif ord("'") == key:
                trig_level += 0.02
            elif ord(";") == key:     # letter p
                trig_level -= 0.02
        except:
            pass
    return


'''
# Trying to make a 'knobs' GUI but OpenCV is really primitive.
# Hence all the exec() statements in what follows...
'''
def set_knobs():
    global knob_names, knobs_nn
    setstr = 'global knobs_nn;  knobs_nn = np.array(['
    num_knobs = len(knob_names)
    for i in range(len(knob_names)):
        knob_name = knob_names[i]
        setstr += f"{knob_name}"
        if i < num_knobs-1: setstr += ','
    setstr += '])'
    #print('setstr = ',setstr)
    exec(setstr)
    #knobs_wc = knobs_nn * ()
    knobs_wc = knob_ranges[:,0] + (knobs_nn+0.5)*(knob_ranges[:,1]-knob_ranges[:,0])

    text = "knobs_wc = "+np.array2string(knobs_wc, precision=3, separator=',',suppress_small=True)
    cv2.rectangle(logo, (0, 0), (500, 25), (255,255,255), -1)
    cv2.putText(logo, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), lineType=cv2.LINE_AA)
    cv2.imshow(knob_controls_window, logo)



def make_controls(cp_items):
    global knob_ranges, knobs_nn, knob_names
    # cp_items are items in the checkpoint file
    for key, value in cp_items:
        if key == 'knob_names':
            knob_names = value
        elif key == 'knob_ranges':
            knob_ranges = np.array(value)
    print("knob_names = ",knob_names)
    print("knob_ranges = ",knob_ranges)

    knobs_nn = np.zeros(len(knob_names))



    num_knobs = len(knob_names)
    for i in range(num_knobs):
        knob_name = knob_names[i]
        exec(f"global {knob_name}; {knob_name} = 0.0")
        fn_name = f"on_{knob_name}_change"
        exec(f'def {fn_name}(val): global {knob_name};  {knob_name} = val/100.0-0.5; set_knobs()')
        exec(f"cv2.createTrackbar(knob_name, knob_controls_window, 50, 100, {fn_name})")
    set_knobs()
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualizes audio activations and neural network weights",\
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', help='Name of model checkpoint .tar file', default="../demo/modelcheckpoint.tar")
    args = parser.parse_args()

    model, cp_items = load_model(args.model)
    model = model.double()

    make_controls(cp_items)

    draw_weights(model)   # show the model weights

    # Call the oscilloscope in order to visualize activations
    scope(model, buf_size=model.in_chunk_size)


# EOF

'''
Demo of compressor using Bokeh server, based on Bokeh example sliders.py

To run, first install Bokeh ("conda install bokeh"), then execute
   bokeh serve bokeh_sliders.py
'''
import numpy as np
import torch
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput, Select, Dropdown
from bokeh.plotting import figure
import os, sys
sys.path.append(os.path.abspath('../'))  # for running from signaltrain/demo/
import signaltrain as st
from signaltrain.nn_modules import nn_proc

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')


# default data settings
chunk_size = 8192*2
sr = 44100


def get_input_sample(chooser, in_chunk_size=8192):
    # input selection
    global t, sr
    t = np.linspace(0,1, in_chunk_size)
    if 'sine' == chooser:
        return st.audio.randsine(t,freq_range=[5,20])
    elif 'box' == chooser:
        return st.audio.box(t)
    elif 'noisy sine' == chooser:
        return st.audio.randsine(t,freq_range=[5,20]) + 0.1*(2*np.random.rand(t.shape[0])-1)
    elif 'noisybox' == chooser:
        return st.audio.box(t) * (2*np.random.rand(t.shape[0])-1)
    elif 'pluck' == chooser:
        return st.audio.pluck(t)
    elif 'real audio' == chooser:
        x =  next(ra_gen)
        t = np.linspace(0,x.shape[0]/sr,x.shape[0])
        return x


def torch_chunkify(x, chunk_size=chunk_size):
    # pads x with zeros and returns a 2D array
    '''
    rows = int(np.ceil(x.shape[0]/chunk_size))  # this will be the batch size
    nearest_mult = rows*(chunk_size)
    xnew = np.zeros(nearest_mult)
    xnew[0:x.shape[0]] = x[0:x.shape[0]]
    xnew  = xnew.reshape(rows, chunk_size)
    '''
    xnew  = x.reshape(1, chunk_size)
    x_torch = torch.autograd.Variable(torch.from_numpy(xnew).to(device), requires_grad=False).float()
    return x_torch



# Set up effect(s)
effects_dict = dict()
effects_dict['comp_4c'] = {'effect':st.audio.Compressor_4c(), 'checkpoint':'modelcheckpoint_16k8k_comp4c.tar'}
effects_dict['denoise'] = {'effect':st.audio.Denoise(), 'checkpoint':'modelcheckpoint_denoise.tar'}
shortname = 'comp_4c'
effect = effects_dict[shortname]['effect']
checkpoint_file = effects_dict[shortname]['checkpoint']
num_knobs = len(effect.knob_names)
knobranges = effect.knob_ranges
knobs_wc = np.array([knobranges[k][0] for k in range(num_knobs)])

# set up model
model = nn_proc.st_model(scale_factor=2, shrink_factor=2, num_knobs=num_knobs, sr=sr)
chunk_size, out_chunk_size = model.in_chunk_size, model.out_chunk_size
state_dict, rv = st.misc.load_checkpoint(checkpoint_file, fatal=True, device="cpu")
if state_dict != {}:
    model.load_state_dict(state_dict)


# input signal
chooser = "box"
show_size = chunk_size//2
x = get_input_sample(chooser, in_chunk_size=show_size)
x = np.concatenate( (np.zeros(len(x)),x))  # double it and add zeros

# target output
y, x = effect.go_wc(x, knobs_wc)

# predicted output
# use the same knob settings for all chunks
x_torch = torch_chunkify(x)   # break up input audio into chunks
knobs_nn = (knobs_wc - knobranges[:,0])/(knobranges[:,1]-knobranges[:,0]) - 0.5
knobs = knobs_nn# np.array([thresh_nn, ratio_nn, attack_nn])
rows = x_torch.size()[0]
knobs = np.tile(knobs,(rows,1))
knobs_torch = torch.autograd.Variable(torch.from_numpy(knobs).to(device), requires_grad=False).float()
y_pred_torch, mag, mag_hat = model.forward(x_torch, knobs_torch)
y_pred = y_pred_torch.data.cpu().numpy().flatten()[0:t.shape[0]]  #flattened numpy version




# Set up plot
source = ColumnDataSource(data=dict(x=t[-show_size:], y=x[-show_size:]))
source2 = ColumnDataSource(data=dict(x=t[-show_size:], y=y[-show_size:]))
source3 = ColumnDataSource(data=dict(x=t[-len(y_pred):], y=y_pred))

plot = figure(plot_height=400, plot_width=400, title="sample output",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, 1], y_range=[-1,1])

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6, legend="Input")
plot.line('x', 'y', source=source2, line_width=3, line_alpha=0.6, color="red", legend="Target")
plot.line('x', 'y', source=source3, line_width=3, line_alpha=0.6, color="green", legend="Predicted")


# Set up widgets
effect_select = Select(title="Effect:", value="box", options=['comp_4c']) # TODO: add 'denoise' later
input_select = Select(title="Signal Type:", value="box", options=['box','sine','pluck','noisybox','noisy sine']) #TODO: add ,'real audio'])
text = TextInput(title="Effect", value=effect.name)
text2 = TextInput(title="Checkpoint", value=checkpoint_file)
knob_sliders = []
for k in range(num_knobs):
    start, end = effect.knob_ranges[k][0], effect.knob_ranges[k][1]
    mid = (start+end)/2
    step = (end-start)/25
    tmp = Slider(title=effect.knob_names[k], value=mid, start=start, end=end, step=step)
    knob_sliders.append(tmp)


# Set up callbacks
def update_data(attrname, old, new):
    global t, x

    # Get the current slider values
    knobs_wc = []
    for k in range(num_knobs):
        knobs_wc.append( knob_sliders[k].value)

    y, _ = effect.go_wc(x, knobs_wc) # generate the new curve
    source2.data = dict(x=t[-show_size:], y=y[-show_size:])

    # call the network in inference mode
    x_torch = torch_chunkify(x)

    knobs_nn = (knobs_wc - knobranges[:,0])/(knobranges[:,1]-knobranges[:,0]) - 0.5
    knobs = knobs_nn# np.array([thresh_nn, ratio_nn, attack_nn])
    rows = x_torch.size()[0]
    knobs = np.tile(knobs,(rows,1))
    knobs_torch = torch.autograd.Variable(torch.from_numpy(knobs).to(device), requires_grad=False).float()
    y_pred_torch, mag, mag_hat = model.forward(x_torch, knobs_torch)
    y_pred = y_pred_torch.data.cpu().numpy().flatten()[0:t.shape[0]]  #flattened numpy version
    source3.data = dict(x=t[-len(y_pred):], y=y_pred)


def update_effect(attrname, old, new):
    shortname = effect_select.value
    effect = effects_dict[shortname]['effect']
    checkpoint_file = effects_dict[shortname]['checkpoint']
    update_data(attrname, old, new)

def update_input(attrname, old, new):
    global x, x_torch
    chooser = input_select.value
    x = get_input_sample(chooser)
    x = np.concatenate( (np.zeros(len(x)),x))  # double it and add zeros

    x_torch = torch_chunkify(x)
    source.data = dict(x=t[-show_size:], y=x[-show_size:])
    update_data(attrname, old, new)

input_select.on_change('value', update_input)
effect_select.on_change('value', update_effect)
for w in knob_sliders:
    w.on_change('value', update_data)



# Set up layouts and add to document
inputs = column([effect_select, input_select]+knob_sliders )

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Sliders"

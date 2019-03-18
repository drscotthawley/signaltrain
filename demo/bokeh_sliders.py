'''
Demo of compressor using Bokeh server, based on Bokeh example sliders.py

To run, first install Bokeh ("conda install bokeh"), then execute
   bokeh serve bokeh_sliders.py
'''
import numpy as np
import torch
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Legend
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


from bokeh.embed import server_document
script = server_document("http://localhost:5006/bokeh_sliders")

print("script =\n",script)

# define default global variables
chunk_size=None
sr = None
knob_names, knob_ranges, num_knobs, knob_sliders = None, None, None, None
chunk_size, out_chunk_size = None, None


def get_input_sample(chooser, in_chunk_size=8192):
    # input selection
    global t, sr
    t = np.linspace(0,1, in_chunk_size)
    if 'sine' == chooser:
        return st.audio.randsine(t,freq_range=[5,20])
    elif 'box' == chooser:
        return st.audio.box(t, delta=0)
    elif 'noisy sine' == chooser:
        return st.audio.randsine(t,freq_range=[5,20]) + 0.1*(2*np.random.rand(t.shape[0])-1)
    elif 'box * noise' == chooser:
        return st.audio.box(t) * (2*np.random.rand(t.shape[0])-1)
    elif 'box + noise' == chooser:
        return st.audio.box(t) + 0.5*np.random.rand()*(2*np.random.rand(t.shape[0])-1)
    elif 'pluck' == chooser:
        return st.audio.pluck(t)
    elif 'real audio' == chooser:
        x =  next(ra_gen)
        t = np.linspace(0,x.shape[0]/sr,x.shape[0])
        return x


def torch_chunkify(x, chunk_size=4096):
    # For long signals.  pads x with zeros and returns a 2D array
    ''' # for now, make it just one big long chunk.
    rows = int(np.ceil(x.shape[0]/chunk_size))  # this will be the batch size
    nearest_mult = rows*(chunk_size)
    xnew = np.zeros(nearest_mult)
    xnew[0:x.shape[0]] = x[0:x.shape[0]]
    xnew  = xnew.reshape(rows, chunk_size)
    '''
    xnew  = x.reshape(1, chunk_size)
    x_torch = torch.autograd.Variable(torch.from_numpy(xnew).to(device), requires_grad=False).float()
    return x_torch

# reads checkpoint file
def setup_model(checkpoint_file, fatal=True):
    global knob_names, knob_ranges, num_knobs, sr, chunk_size, out_chunk_size
    state_dict, rv = st.misc.load_checkpoint(checkpoint_file, fatal=fatal, device="cpu")
    if {} == state_dict:
        return None
    scale_factor = rv['scale_factor']
    shrink_factor = rv['shrink_factor']
    knob_names = rv['knob_names']
    knob_ranges = rv['knob_ranges']
    num_knobs = len(knob_names)
    sr = rv['sr']
    # set up model
    model = nn_proc.st_model(scale_factor=scale_factor, shrink_factor=shrink_factor, num_knobs=num_knobs, sr=sr)
    model.load_state_dict(state_dict)  # overwrite weights using checkpoint info
    chunk_size, out_chunk_size = model.in_chunk_size, model.out_chunk_size
    return model

# Set up list of effects. written generally so we can add other effects later
effects_dict = dict()
effects_dict['comp_4c'] = {'name':'Comp-4c: 4-Knob Compressor', 'effect':st.audio.Compressor_4c(), 'checkpoint':'model_comp4c_4k.tar'}
# other effects to enable later:
#effects_dict['comp_3c'] = {'name':'3-Knob Compressor', 'effect':st.audio.Compressor(),    'checkpoint':'model_comp3c_4k.tar'}
effects_dict['denoise'] = {'name':'Extra: (Tunable) Denoiser',          'effect':st.audio.Denoise(),      'checkpoint':'modelcheckpoint_denoise.tar'} # don't link in audio.Denoise()
#effects_dict['decomp_4c'] = {'name':'4-Knob De-Compressor', 'effect':None, 'checkpoint':''} # do not try to use decompressor effect
#effects_dict['nothing'] = {'name':'Nothing (for testing)', 'effect':None,  'checkpoint':''}
shortname = 'comp_4c'   # select default effect

# read and parse the checkpoint fileself
#  Note that this will also set a ton of globals, e.g. chunk_size
checkpoint_file = effects_dict[shortname]['checkpoint']
model = setup_model(checkpoint_file)

effect = effects_dict[shortname]['effect']  # set up the call for target data


#---------  Initial data for the plot.  this will all need to be updated later...
# input signal
chooser = "box"
show_size = out_chunk_size
x = get_input_sample(chooser, in_chunk_size=chunk_size)
#x = np.concatenate( (np.zeros(len(x)),x))  # double it and add zeros

# default outputs just in case the effect or model are undefined
y = 0*x
y_pred = x[0:1]

# target output
knobs_wc = knob_ranges.mean(axis=1)
if effect is not None:
    y, x = effect.go_wc(x, knobs_wc)

# predicted output.     use the same knob settings for all chunks
if model is not None:
    x_torch = torch_chunkify(x, chunk_size=chunk_size)   # break up input audio into chunks
    knobs_nn = (knobs_wc - knob_ranges[:,0])/(knob_ranges[:,1]-knob_ranges[:,0]) - 0.5
    rows = x_torch.size()[0]
    knobs = np.tile(knobs_nn,(rows,1))
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

plot.line('x', 'y', source=source, line_width=2.5, line_alpha=0.8, line_color="navy", legend="Input")
plot.line('x', 'y', source=source2, line_width=2.5, line_alpha=0.8, line_dash=[4, 4], line_color="orange", legend="Target")
plot.line('x', 'y', source=source3, line_width=2.5, line_alpha=0.6, line_color="green", legend="Predicted")
plot.legend.location = "bottom_left"
plot.legend.background_fill_alpha = 0.5


# Set up widgets
effect_options = [effects_dict[x]['name'] for x in effects_dict.keys()]
effect_select = Select(title="Effect:", value=effect_options[0], options=effect_options)
input_select = Select(title="Input Signal:      (randomly gen'd on change)", value="box", options=['box','sine','pluck','box * noise','noisy sine','box + noise']) #TODO: add ,'real audio'])
knob_sliders = []
for k in range(num_knobs):
    start, end = knob_ranges[k][0], knob_ranges[k][1]
    mid = (start+end)/2
    step = (end-start)/25
    tmp = Slider(title=knob_names[k], value=mid, start=start, end=end, step=step)
    knob_sliders.append(tmp)


# Set up callbacks
def update_data(attrname, old, new):
    global t, x, y
    plot.title.text = "sample data"

    source.data = dict(x=t[-show_size:], y=x[-show_size:])

    # Get the current slider values
    knobs_wc = []
    if num_knobs > 0 :
        for k in range(num_knobs):
            knobs_wc.append( knob_sliders[k].value)

    # generate the new target curve, if possible
    if (effect is not None) and ('De' not in effect.name): # don't update x for de- effects. (even though we should; but doing so makes the demo behave funny)
        y, x_tmp = effect.go_wc(x, knobs_wc)
        source2.data = dict(x=t[-show_size:], y=y[-show_size:])
    else:
        source2.data = dict(x=t[0:1], y=0*t[0:1])  # effectively remove the display of data for this one
        plot.title.text += ", no target"

    # call the model in inference mode
    if model is not None:
        x_torch = torch_chunkify(x, chunk_size=len(t))
        knobs_nn = (knobs_wc - knob_ranges[:,0])/(knob_ranges[:,1]-knob_ranges[:,0]) - 0.5
        knobs = knobs_nn# np.array([thresh_nn, ratio_nn, attack_nn])
        rows = x_torch.size()[0]
        knobs = np.tile(knobs,(rows,1))
        knobs_torch = torch.autograd.Variable(torch.from_numpy(knobs).to(device), requires_grad=False).float()
        y_pred_torch, mag, mag_hat = model.forward(x_torch, knobs_torch)
        y_pred = y_pred_torch.data.cpu().numpy().flatten()[0:t.shape[0]]  #flattened numpy version
        source3.data = dict(x=t[-len(y_pred):], y=y_pred)
    else:
        source3.data = dict(x=t[0:1], y=0*t[0:1])
        plot.title.text += ", no predicted"



def update_effect(attrname, old, new):
    global effect, model, knob_names, knob_ranges, num_knobs, knob_sliders
    # match the menu option with the right entry in effects_dict
    long_name = effect_select.value
    plot.title.text = f"Trying to setup effect '{long_name}'..."
    shortname = ''
    for key, val in effects_dict.items():
        if val['name'] == long_name:
            shortname = key
            break
    if '' == shortname:
        plot.title.text = f"**ERROR: Effect '{long_name}' not defined**"
        return
    effect = effects_dict[shortname]['effect']
    num_knobs = 0
    if effect is not None:
        knob_names, knob_ranges = effect.knob_names, np.array(effect.knob_ranges)
        num_knobs = len(knob_names)

    # try to read the checkpoint file
    checkpoint_file = effects_dict[shortname]['checkpoint']
    model = setup_model(checkpoint_file, fatal=False)
    if model is  None:
        msg = f"**ERROR: checkpoint file '{checkpoint_file}' not found**"
        print("\n",msg)
        plot.title.text = msg

    # rebuild the entire display  (because knobs have changed)
    knob_sliders = []
    if num_knobs > 0:
        knobs_wc = knob_ranges.mean(axis=1)
        for k in range(num_knobs):
            start, end = knob_ranges[k][0], knob_ranges[k][1]
            mid = knobs_wc[k]
            step = (end-start)/25
            tmp = Slider(title=knob_names[k], value=mid, start=start, end=end, step=step)
            knob_sliders.append(tmp)
        for w in knob_sliders:  # since we now defined new widgets, we need triggers for them
            w.on_change('value', update_data)

    inputs = column([effect_select, input_select]+knob_sliders )
    curdoc().clear()
    curdoc().add_root(row(inputs, plot, width=800))
    curdoc().title = "SignalTrain Demo"

    update_data(attrname, old, new)


def update_input(attrname, old, new):
    global t, x, x_torch, chunk_size
    chooser = input_select.value
    x = get_input_sample(chooser, in_chunk_size=chunk_size)
    #x = np.concatenate( (np.zeros(len(x)),x))  # double it and add zeros

    x_torch = torch_chunkify(x, chunk_size=chunk_size)
    source.data = dict(x=t[-show_size:], y=x[-show_size:])
    update_data(attrname, old, new)


# watch for changes and call callbacks
input_select.on_change('value', update_input)
effect_select.on_change('value', update_effect)
for w in knob_sliders:
    w.on_change('value', update_data)

# Set up layouts and add to document
inputs = column([effect_select, input_select]+knob_sliders )
curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "SignalTrain Demo"

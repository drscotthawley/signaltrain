#! /usr/bin/env python3
"""
predicts a single long segment of audio by repeatedly predicting chunks and putting
them together

Can be run as a standalone utility routine, or as a function called from another method

"""

import numpy as np
import torch
import sys
sys.path.append('/home/shawley/signaltrain')
sys.path.append('..')
import signaltrain as st
import glob

# NVIDIA Apex for mixed-precision training
have_apex = False
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
    have_apex = True
except ImportError:
    print("Recommend that you install apex from https://www.github.com/nvidia/apex to run this code.")


def predict_long(signal, knobs_nn, model, chunk_size, out_chunk_size, sr=44100, effect=None, device="cpu:0"):

    # reshape input and knobs.  break signal up into overlapping windows
    overlap = chunk_size-out_chunk_size
    print("predict_long: chunk_size, out_chunk_size, overlap = ",chunk_size, out_chunk_size, overlap)
    x = st.audio.sliding_window(signal, chunk_size, overlap=overlap)
    print("predict_long: x.shape, signal.shape = ",x.shape, signal.shape )
    batch_size = x.shape[0]
    if x.shape[0] > 200:
        print(f"**WARNING: effective batch size = {x.shape[0]}, may be too large and produce CUDA out of memory errors")
        batch_size = 200

    knobs = np.tile(knobs_nn, (batch_size,1))     # repeat knob settings a bunch of times
    knobs_torch = torch.Tensor(knobs.astype(np.float32, copy=False)).to(device)

    y_pred = np.empty( shape=(0) )
    # move through the audio file, one batch at a time
    bmax = int(np.round(x.shape[0]/batch_size))
    print("bmax = ",bmax)
    for b in range(bmax):
        print('batch id b =',b,end="")
        bstart = b*batch_size
        if b == bmax-1:   # last batch
            batch_size = x.shape[0] - bstart
        print(', bstart = ',bstart,', batch_size = ',batch_size)
        # Move data to torch device
        knobs = np.tile(knobs_nn, (batch_size,1))     # repeat knob settings a bunch of times
        knobs_torch = torch.Tensor(knobs.astype(np.float32, copy=False)).to(device)
        x_torch = torch.Tensor(x[bstart:bstart+batch_size].astype(np.float32, copy=False)).to(device)

        # Do the model prediction
        y_hat, mag, mag_hat = model.forward(x_torch, knobs_torch)

        # Reassemble the output into one long signal
        # Note: we don't need (or want) to undo_sliding_window() because y_hat has no overlaps
        y_pred = np.append( y_pred, y_hat.squeeze(0).cpu().detach().numpy().flatten().astype(np.float32,copy=False) )

    # note that sliding_window() probably tacked zeros onto the end, so let's remember to take them off
    unique = x.shape[1] + (x.shape[0]-1)*(x.shape[1]-overlap) # number of unique values in windowed x (including extra zeros but not overlaps)
    num_extra = unique - signal.size                          # difference between that and original signal
    print("predict_long:  y_pred.shape, num_extra = ",y_pred.shape, num_extra)
    if num_extra > 0:
        return y_pred[0:-num_extra]
    else:
        return y_pred


def calc_ct(signal, effect, knobs_wc, out_chunk_size, chunk_size, sr=44100):
    # calculate chunked target audio
    lookback_size = chunk_size - out_chunk_size
    if lookback_size >= 0:
        padded_sig = np.concatenate((np.zeros(lookback_size, dtype=np.float32), signal))
        y_ct = np.zeros(len(padded_sig))                      # start with y_ct all zeros
        for i in np.arange(0, len(padded_sig), out_chunk_size):
            iend = min( i+chunk_size, len(padded_sig))        # where's the end of this
            in_chunk = padded_sig[i:iend]                     # grab input chunk from padded signal
            out_chunk, _ = effect.go_wc(in_chunk, knobs_wc)   # apply effect on this chunk
            if len(out_chunk) > out_chunk_size:               # watch out for array sizes...
                out_chunk = out_chunk[-out_chunk_size:]
            itbgn, itend = iend - len(out_chunk), iend
            y_ct[itbgn:itend] = out_chunk                     # paste the result into y_ct
        y_ct = y_ct[lookback_size:]                           # remove padding
    return y_ct



if __name__ == "__main__":
    ## Can be run as standalone app for testing / eval purposes
    import os
    import argparse
    from signaltrain.nn_modules import nn_proc

    # torch device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type('torch.FloatTensor')

    # parse command line args
    parser = argparse.ArgumentParser(description="Runs NN inference on long audio clip",\
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('checkpoint', help='Name of model checkpoint .tar file')
    parser.add_argument('audiofile', help='Name of audio file to read')
    parser.add_argument('--effect', help='Name of effect class for generating target', default='comp_4c')
    parser.add_argument('--knobs', help='String of knob/control settings', default='-30.0, 5.0, 0.04, 0.04')
    # info is in checkpoint file  parser.add_argument('--path', help='Path from which to load file-based effects', default='')
    args = parser.parse_args()
    print("args =",args)

    # load from checkpoint
    print("Looking for checkpoint at",args.checkpoint)
    state_dict, rv = st.misc.load_checkpoint(args.checkpoint, fatal=True)
    scale_factor, shrink_factor = rv['scale_factor'], rv['shrink_factor']
    knob_names, knob_ranges = rv['knob_names'], rv['knob_ranges']
    num_knobs = len(knob_names)
    sr = rv['sr']
    chunk_size, out_chunk_size = rv['in_chunk_size'], rv['out_chunk_size']
    print(f"Effect name = {rv['effect_name']}")
    print(f"knob_names = {knob_names}")
    print(f"knob_ranges = {knob_ranges}")

    # Setup model
    model = nn_proc.st_model(scale_factor=scale_factor, shrink_factor=shrink_factor, num_knobs=num_knobs, sr=sr)
    model.load_state_dict(state_dict)   # overwrite the weights using the checkpoint
    chunk_size = model.in_chunk_size
    out_chunk_size = model.out_chunk_size
    print("out_chunk_size = ",out_chunk_size)


    if have_apex:
        optimizer = torch.optim.Adam(list(model.parameters()))
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")


    # Input Data
    #infile="/home/shawley/datasets/signaltrain/music/Test/WindyPlaces.ITB.Mix10-2488-1644.wav"
    infile = args.audiofile
    print("reading input file ",infile)
    signal, sr = st.audio.read_audio_file(infile, sr=sr)
    print("signal.shape = ",signal.shape)
    y_ct = None

    ##### KNOB SETTINGS HERE
    # Can hard code them here or, for 'files' effects, they're inferred from the target (below)
    #knobs_wc = np.array([-30, 2.5, .002, .03])  # 4-knob compressor settings, for Windy Places in demo
    #knobs_wc = np.array([-20, 5, .01, .04])  # 4-knob compressor settings, for Leadfoot in demo
    #knobs_wc = np.array([-40])  # comp with only 1 knob 'thresh'
    #knobs_wc = np.array([1,85])
    #knobs_wc = np.array([-30.0, 5.0, 0.04, 0.04])
    knobs_wc = np.fromstring(args.knobs, dtype=np.float32, sep=',')
    print("default knobs_wc  =",knobs_wc)


    # Generate Target audio  - in one big stream: "streamed target" data
    print("\nhello. args.effect=",args.effect)
    do_target = (args.effect != '')
    if do_target:
        if args.effect == 'comp_4c':
            effect = st.audio.Compressor_4c()
        elif args.effect == 'comp_4c_large':
            effect = st.audio.Compressor_4c_Large()
        elif args.effect == 'comp_t':
            effect = st.audio.Comp_Just_Thresh()
        elif args.effect == 'files':
            print('going to try to load what we can')
            #target_file = '/home/shawley/datasets/LA2A_LC_032019/Val/target_218_LA2A_3c__1__85.wav'
            # use the input filename and the knob vals to get the target val
            target_file = infile.replace('input','target').replace('.wav','')
            target_file = glob.glob(target_file+"*")[0]
            print(" Reading target_file = ",target_file)
            y_st, _ = st.audio.read_audio_file(target_file)
            print("-------------------------------   len(y_st) = ",len(y_st))
            # get the knob settings from the corresponding target filename
            subs = target_file.replace('.wav','').split('__')
            knobs_wc = [np.float(x) for x in subs[1:]]
            knobs_wc = np.array(knobs_wc)
            print("inferred knobs_wc = ",knobs_wc)
        else:
            print("WARNING: That effect not implemented yet. Skipping target generation.")

        if 'comp' in args.effect:
            y_st, _ = effect.go_wc(signal, knobs_wc)
            y_ct = calc_ct(signal, effect, knobs_wc, out_chunk_size, chunk_size)

    # convert to NN parameters for knobs
    kr = np.array(knob_ranges)
    knobs_nn = (knobs_wc - kr[:,0])/(kr[:,1]-kr[:,0]) - 0.5

    # Call the predict_long routine
    print("\nCalling predict_long()...")
    y_pred = predict_long(signal, knobs_nn, model, chunk_size, out_chunk_size, sr=sr, device=device)

    print("\n...Back. Output: y_pred.shape = ",y_pred.shape)

    if (do_target):
        print("y_st.shape = ",y_st.shape)
        print("diff in lengths = ",len(y_st)-len(y_pred))

    # output files (offset pred with zeros to time-match with input & target)
    y_out = np.zeros(len(y_st),dtype=np.float32)
    y_out[-len(y_pred):] = y_pred

    print("Output y_out.shape = ",y_out.shape)

    # write output files, which have been properly aligned
    tagstr = ''
    for i in range(len(knobs_wc)):
        tagstr += '__'+str((knobs_wc[i]))
    st.audio.write_audio_file("pl_input"+tagstr+".wav", signal, sr=44100)
    st.audio.write_audio_file("pl_pred"+tagstr+".wav", y_out, sr=44100)
    if do_target:
        st.audio.write_audio_file("pl_st"+tagstr+".wav", y_st, sr=44100)
        if y_ct is not None:
            st.audio.write_audio_file("pl_ct"+tagstr+".wav", y_ct, sr=44100)

    print("Finished.")

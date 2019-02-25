"""
predicts a single long segment of audio
"""

import numpy as np
import torch
import sys
sys.path.append('..')
import signaltrain as st

def predict_long(signal, knobs_nn, model, chunk_size, out_chunk_size, sr=44100, effect=None):

    # reshape input and knobs.  break signal up into overlapping windows
    overlap = chunk_size-out_chunk_size
    print("overlap = ",overlap)
    x = st.audio.sliding_window(signal, chunk_size, overlap=overlap)
    knobs = np.tile(knobs_nn, (x.shape[0],1))     # repeat knob settings a bunch of times

    # Move data to torch device
    x, knobs = torch.Tensor(x.astype(np.float32)), torch.Tensor(knobs.astype(np.float32))
    print("x.size() =",x.size(), ", knobs.size() =",knobs.size() )
    x_cuda, knobs_cuda = x.to(device),  knobs.to(device)

    # Do the model prediction
    y_hat, mag, mag_hat = model.forward(x_cuda, knobs_cuda)

    # Reassemble the output into one long signal
    y_pred = y_hat.cpu().detach().numpy().flatten().astype(np.float32)

    return y_pred


if __name__ == "__main__":
    import os
    import argparse
    from signaltrain.nn_modules import nn_proc

    # load from checkpoint
    # TODO: replace hard-coded name with command-line argument 
    checkpointname = '../data/comp_4c/modelcheckpoint.tar'
    if os.path.isfile(checkpointname):
        print("Checkpoint file found. Loading weights.")
        checkpoint = torch.load(checkpointname,) # map_location=device)
    else:
        print("Error, no checkpoint found")
        sys.exit(1)

    # torch device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type('torch.FloatTensor')



    # Setup model
    scale_factor = 1  # change dimensionality of run by this factor
    chunk_size = int(8192 * scale_factor)   # size of audio that NN model expects as input
    output_sf = 4     # output shrink factor, i.e. fraction of output actually trained on
    out_chunk_size = chunk_size // output_sf        # size of output audio that we actually care about
    num_knobs = 4   # len(checkpoint['knob_names'])
    sr = 44100

    print("Input chunk size =",chunk_size)
    print("Output chunk size =",out_chunk_size)
    print("Sample rate =",sr)

    # Analysis parameters
    ft_size = int(1024 * scale_factor)
    hop_size = int(384 * scale_factor)
    expected_time_frames = int(np.ceil(chunk_size/float(hop_size)) + np.ceil(ft_size/float(hop_size)))
    output_time_frames = int(np.ceil(out_chunk_size/float(hop_size)) + np.ceil(ft_size/float(hop_size)))
    y_size = (output_time_frames-1)*hop_size - ft_size

    # Initialize Model
    model = nn_proc.AsymMPAEC(expected_time_frames, ft_size=ft_size, hop_size=hop_size, n_knobs=num_knobs, output_tf=output_time_frames)
    model.load_state_dict(checkpoint['state_dict'])


    # Input Data
    # TODO: replace hard-coding with command-line argument
    infile="/home/shawley/datasets/signaltrain/music/Test/WindyPlaces.ITB.Mix10-2488-1644.wav"
    signal, sr = st.audio.read_audio_file(infile, sr=sr)
    print("signal.shape = ",signal.shape)

    # Knob settings
    knobs_nn = np.random.rand(num_knobs)-0.5
    knobs_nn[0] = -0.4  # crank down the threshold for testing

    # Target audio
    effect = st.audio.Compressor_4c()
    y, _ = effect.go(signal, knobs_nn)

    # Call the predict_long routine
    y_pred = predict_long(signal, knobs_nn, model, chunk_size, out_chunk_size, sr=sr)

    print("Output y_pred.shape = ",y_pred.shape)
    #y_pred = st.audio.undo_sliding_window(y_pred, overlap)

    # output files (offet pred with zeros to time-match with input & target)
    y_out = np.zeros(len(signal),dtype=np.float32)
    y_out[-len(y_pred):] = y_pred
    st.audio.write_audio_file("input.wav", signal, sr=44100)
    st.audio.write_audio_file("y_pred.wav", y_out, sr=44100)
    st.audio.write_audio_file("y_target.wav", y, sr=44100)

    print("Finished.")

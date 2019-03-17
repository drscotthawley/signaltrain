#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Scott H. Hawley'

# imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os, sys
import time
from signaltrain import audio, io_methods, learningrate, datasets, loss_functions, misc
from signaltrain.nn_modules import nn_proc


try:   # NVIDIA Apex for mixed-precision training
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this code.")



def train(effect=audio.Compressor_4c(), epochs=100, n_data_points=200000, batch_size=20,
    device=torch.device("cuda:0"), plot_every=10, cp_every=25, sr=44100, datapath=None,
    scale_factor=1, shrink_factor=4, synth_prob=0.5):
    """
    Main training routine for signaltrain

    Parameters:
        effect:           class for the audio effect to learn (see audio.py)
        epochs:           how many epochs to run over
        n_data_points:    data instances per epoch (or iterations per epoch)
        batch_size:       batch size
        device:           pytorch device to run on, either cpu or cuda (GPU)
        plot_every:       how often to generate plots of sample outputs
        cp_every:         save checkpoint every this many iterations
        scale_factor:     change overal dimensionality of i/o chunks by this factor
        shrink_factor:    output shrink factor, i.e. fraction of output actually trained on
        synth_prob:       TODO: unused. might use it to combine on-the-fly data-gen with files
    """

    # print info about this training run
    print(f'SignalTrain training execution began at {time.ctime()}. Options:')
    print(f'    epochs = {epochs}, n_data_points = {n_data_points}, batch_size = {batch_size}')
    print(f'    scale_factor = {scale_factor}, shrink_factor = {shrink_factor}')
    num_knobs = len(effect.knob_names)
    print(f'    num_knobs = {num_knobs}')
    effect.info()  # Print effect settings

    # Setup the Model
    #model = nn_proc.AsymMPAEC(expected_time_frames, ft_size=ft_size, hop_size=hop_size, n_knobs=num_knobs, output_tf=output_time_frames)
    model = nn_proc.st_model(scale_factor=scale_factor, shrink_factor=shrink_factor, num_knobs=num_knobs, sr=sr)
    chunk_size, out_chunk_size = model.in_chunk_size, model.out_chunk_size
    y_size = out_chunk_size
    print("Model defined.  Number of trainable parameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.to(device)

    # Specify learning rate schedule...although we don't bother stepping the momentum
    lr_max = 1.0e-4                         # obtained after running helpers.learningrate.lr_finder.py
    lr_sched = learningrate.get_1cycle_schedule(lr_max=lr_max, n_data_points=n_data_points, epochs=epochs, batch_size=batch_size)
    maxiter = len(lr_sched)
    eps_sched = 1e-8 + 0.5-0.5*np.cos(np.linspace(0, np.pi, maxiter))

    # Initialize optimizer. given our "random" training data, weight decay seem to doesn't help but rather slows training way down
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr_sched[0], weight_decay=0)


    # Are we synthesizing data or do we expect it to come from files
    synth_data = datapath is None

    # Setup/load data
    if synth_data:  # synthesize data  # TODO: warning: broke this a while ago in favor of using files.
        dataset = datasets.SynthAudioDataSet(chunk_size, effect, sr=sr, datapoints=n_data_points, y_size=out_chunk_size)
        dataset_val = datasets.SynthAudioDataSet(chunk_size, effect, sr=sr, datapoints=n_data_points//4, recycle=True, y_size=out_chunk_size)
    else: # use files
        dataset = datasets.AudioFileDataSet(chunk_size, effect, sr=sr,  datapoints=n_data_points, path=datapath+"/Train/",  y_size=out_chunk_size, rerun=False)
        dataset_val = datasets.AudioFileDataSet(chunk_size, effect, sr=sr, datapoints=n_data_points//4, path=datapath+"/Val/", y_size=out_chunk_size, rerun=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=10, shuffle=True, worker_init_fn=datasets.worker_init) # need worker_init for more variance
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=10, shuffle=False)


    # Load from checkpoint if it exists
    checkpointname = 'modelcheckpoint.tar'
    state_dict, rv = misc.load_checkpoint(checkpointname, fatal=False)
    if state_dict != {}:
        model.load_state_dict(state_dict)
        # TODO: set optimizer state from checkpoint


    # mixed precision
    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2")


    # Copy model to (other) GPU if possbible
    parallel = False # torch.cuda.device_count() > 1
    if parallel:       # For Hawley's 2 Titan X GPUs this typically cuts execution time down by ~30% (not 50%)
        print("Replicating NN model for data-parallel execution across", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    #model = model.half()
    #model.to(device)             # put the model on the GPU if it wasn't already

    # Set up for using additional L1 regularization - scale by frequency bin
    scale_by_freq = None

    # Setup log file
    logfilename = "vl_avg_out.dat"   # Val Loss, average, output
    with open(logfilename, "w") as myfile:  # save progress of val loss
        myfile.close()


    # Loop over epochs
    iter_count, batch_num, status_every = 0, 0, 10   # some loop-related meta-variabled initializations
    avg_loss, vl_avg, beta = 0.0, 0.0, 0.98  # for reporting, average over last 50 iterations
    first_time = time.time()                 # start the clock for logging purposes
    for epoch in range(epochs):
        print("")
        data_point=0                         # within this epoch, count which 'data point' we're using (i.e. which audio sample is at the beginning of our batch)

        # Training phase
        model.train()
        for x, y, knobs in dataloader:
            # move to GPU TODO: does DataLoader already do this?
            x_cuda, y_cuda, knobs_cuda = x.to(device), y.to(device), knobs.to(device)

            lr = lr_sched[min(iter_count, len(lr_sched)-1)]  # get value for learning rate
            eps = eps_sched[min(iter_count, len(lr_sched)-1)]
            data_point += batch_size

            # feed-forward synthesis
            y_hat, mag, mag_hat = model.forward(x_cuda, knobs_cuda)

            #print("y_hat, mag, mag_hat = ",y_hat, mag, mag_hat)
            # set up loss weighting
            if scale_by_freq is None:
                expfac = 7./mag_hat.size()[-1]   # exponential L1 loss scaling by ~1000 times (30dB) over freq range
                scale_by_freq = torch.exp(expfac* torch.arange(0., mag_hat.size()[-1])).expand_as(mag_hat)

            # loss evaluation
            loss = loss_functions.calc_loss(y_hat.float(), y_cuda.float(), mag_hat.float(), scale_by_freq=scale_by_freq.float())

            # Status message
            batch_num += 1
            if 0 == batch_num % status_every:
                avg_loss = beta * avg_loss + (1-beta) * loss.item()
                smoothed_loss = avg_loss / (1 - beta**batch_num)
                timediff = time.time() - first_time
                print(f"\repoch {epoch+1}/{epochs}, time: {timediff:.2f}: lr = {lr:.2e}, data_point {data_point}: loss: {smoothed_loss:.3e}   ",end="")

            # Optimization
            optimizer.zero_grad()

            #loss.backward()   # replace this with amp call
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm=1., norm_type=1)

            # Clip model norm
            '''
            if parallel:
                for child in model.children():  # for DataParallel
                    if hasattr(child, 'clip_grad_norm_'):
                        child.clip_grad_norm_()
            else:
                if hasattr(model, 'clip_grad_norm_'):
                    model.clip_grad_norm_(optimizer)   # for non-DataParallel
            '''

            optimizer.step()

            # Apply Learning rate scheduling
            optimizer.param_groups[0]['lr'] = lr     # adjust according to schedule
            #optimizer.param_groups[0]['eps'] = eps
            iter_count += 1
        # end of loop over training batches. Training phase finished for this epoch


        # Validation phase
        model.eval()
        val_batch_num = 0
        for x_val, y_val, knobs_val in dataloader_val:
            val_batch_num += 1
            x_val_cuda, y_val_cuda, knobs_val_cuda = x_val.to(device), y_val.to(device), knobs_val.to(device)

            y_val_hat, mag_val, mag_val_hat = model.forward(x_val_cuda, knobs_val_cuda)
            loss_val = loss_functions.calc_loss(y_val_hat.float(), y_val_cuda, mag_val_hat.float())
            vl_avg = beta*vl_avg + (1-beta)*loss_val.item()    # (running) average val loss
            if 0 == val_batch_num % status_every:
                timediff = time.time() - first_time
                print(f"\repoch {epoch+1}/{epochs}, time: {timediff:.2f}: lr = {lr:.2e}, data_point {data_point}: loss: {smoothed_loss:.3e} val_loss: {vl_avg:.3e}   ",end="")

        #  Write various forms of status output...
        with open(logfilename, "a") as myfile:  # save progress of val loss to text file
            myfile.write(f"{epoch+1} {vl_avg:.3e}\n")

        if (epoch+1) % plot_every == 0:  # plot sample val data
            print("\nSaving sample data plots",end="")
            io_methods.plot_valdata(x_val_cuda, knobs_val_cuda, y_val_cuda, y_val_hat, effect, epoch, loss_val, target_size=y_size)

        if ((epoch+1) % 20 == 0) or (epoch == epochs-1):    # write out spectrograms from time to time
            io_methods.plot_spectrograms(model, mag_val, mag_val_hat)

        # save checkpoint of model to file, which can be loaded later
        if ((epoch+1) % cp_every == 0):
            misc.save_checkpoint(checkpointname, model, epoch, parallel, optimizer, effect, sr)

        # time estimate (got tired of calcuting this by hand every time!)
        if ((epoch+1)==1):
            secs_left = (time.time() - first_time) * (epochs-1)
            future = time.time() + secs_left
            hours =  secs_left / 3600.0
            print(f"\nExpect run to finish in roughly {hours:.1f} hours, on {time.ctime(future)}")

    print("\nTotal elapsed time =",time.time() - first_time)
    return None


if __name__ == "__main__":
    np.random.seed(218)
    torch.manual_seed(218)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(218)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type('torch.FloatTensor')
    train(epochs=1000, n_data_points=200000, batch_size=200, device=device, effect=audio.Compressor_4c())
    #main(epochs=1000, n_data_points=200000, batch_size=200, device=device,effect=audio.LA2A_Files())

# EOF

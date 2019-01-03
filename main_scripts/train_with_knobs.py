#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import time
# import signaltrain routines
sys.path.append(os.path.abspath('../'))
from helpers import audio, io_methods, learningrate, data
from nn_modules import nn_proc
from losses import loss_functions


def main(epochs=100, n_data_points=1, batch_size=20, device=torch.device("cuda:0"), plot_every=10, effect=audio.Compressor_4c()):
    '''
    '''

    # Data settings
    scale_factor = 2  # change dimensionality of run by this factor
    chunk_size = int(8192 * scale_factor)   # size of audio that NN model expects as input
    output_sf = 4     # output shrink factor
    out_chunk_size = chunk_size // output_sf        # size of output audio that we actually care about
    sampling_freq = 44100.
    # Analysis parameters
    ft_size = int(1024 * scale_factor)
    hop_size = int(384 * scale_factor)
    expected_time_frames = int(np.ceil(chunk_size/float(hop_size)) + np.ceil(ft_size/float(hop_size)))
    output_time_frames = int(np.ceil(out_chunk_size/float(hop_size)) + np.ceil(ft_size/float(hop_size)))
    y_size = (output_time_frames-1)*hop_size - ft_size
    print("expected_time_frames =",expected_time_frames,", output_time_frames =",output_time_frames,", y_size =",y_size)
    # print info about this run
    print(f'SignalTrain execution began at {time.time()}. Options:')
    print(f'    epochs = {epochs}, n_data_points = {n_data_points}, batch_size = {batch_size}, ')
    print(f'    scale_factor = {scale_factor}:  chunk_size = {chunk_size}, ft_size = {ft_size}, hop_size = {hop_size}')
    effect.info()  # Print effect settings

    # Initialize nn modules
    model = nn_proc.AsymMPAEC(expected_time_frames, ft_size=ft_size, hop_size=hop_size, n_knobs=len(effect.knob_names), output_tf=output_time_frames)
    print("Model defined.  Number of trainable parameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))
    #model = nn_proc.Ensemble(model, N=4)

    # load from checkpoint
    checkpointname = 'modelcheckpoint.tar'
    if os.path.isfile(checkpointname):
        print("Checkpoint file found. Loading weights.")
        checkpoint = torch.load(checkpointname,) # map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

    parallel = torch.cuda.device_count() > 1
    if parallel:       # For Hawley's 2 GPUs this cuts execution time down by ~30% (not 50%)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)


    # learning rate schedule...although we don't bother stepping the momentum
    lr_max = 1.0e-3                         # obtained after running helpers.learningrate.lr_finder.py
    lrs = learningrate.get_1cycle_schedule(lr_max=lr_max, n_data_points=n_data_points, epochs=epochs, batch_size=batch_size)

    # Initialize optimizer. given our "random" training data, weight decay doesn't help but rather slows training way down
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lrs[0], weight_decay=0)

    # setup log file
    logfilename = "vl_avg_out.dat"
    with open(logfilename, "w") as myfile:  # save progress of val loss
        myfile.close()

    # set up data
    skip_factor = 0# 1 - out_chunk_size/chunk_size
    synth_data = False
    if synth_data:  # synthesize data
        dataset = data.AudioDataGenerator(chunk_size, effect, sr=sampling_freq, datapoints=n_data_points)
        dataset_val = data.AudioDataGenerator(chunk_size, effect, sr=sampling_freq, datapoints=n_data_points//4, recycle=True)
    else: # use files
        dataset = data.AudioFileDataSet(chunk_size, effect, sr=sampling_freq,  datapoints=n_data_points, path="Train/", skip_factor=skip_factor, y_size=y_size,rerun=False)
        dataset_val = data.AudioFileDataSet(chunk_size, effect, sr=sampling_freq, datapoints=n_data_points//4, path="Val/", skip_factor=skip_factor, y_size=y_size,rerun=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=10, shuffle=True, worker_init_fn=data.worker_init) # need worker_init for more variance
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=10, shuffle=False)


    # epochs
    avg_loss, vl_avg, beta = 0.0, 0.0, 0.98  # for reporting, average over last 50 iterations
    iter_count = 0
    first_time = time.time()
    batch_num = 0
    for epoch in range(epochs):
        print("")
        data_point=0

        # Training phase
        model.train()
        for x, y, knobs in dataloader:
            x_cuda, y_cuda, knobs_cuda = x.to(device), y.to(device), knobs.to(device)

            lr = lrs[min(iter_count, len(lrs)-1)]  # get value for learning rate
            data_point += batch_size

            # forward synthesis
            y_hat, mag, mag_hat = model.forward(x_cuda, knobs_cuda)
            loss = loss_functions.calc_loss(y_hat, y_cuda, mag_hat)

            # Status message
            batch_num += 1
            avg_loss = beta * avg_loss + (1-beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            timediff = time.time() - first_time
            print(f"\repoch {epoch+1}/{epochs}, time: {timediff:.2f}: lr = {lr:.2e}, data_point {data_point}: loss: {smoothed_loss:.3e}   ",end="")

            # Optimization
            optimizer.zero_grad()
            loss.backward()
            # Clip norm
            if parallel:
                for child in model.children():  # for DataParallel
                    child.clip_grad_norm_()
            else:
                model.clip_grad_norm_()   # for non-DataParallel
            optimizer.step()

            # Learning rate scheduling
            optimizer.param_groups[0]['lr'] = lr     # adjust according to schedule
            iter_count += 1
        # end of loop over training batches. Training phase finished for this epoch


        # Validation phase
        model.eval()
        for x_val, y_val, knobs_val in dataloader_val:
            x_val_cuda, y_val_cuda, knobs_val_cuda = x_val.to(device), y_val.to(device), knobs_val.to(device)

            y_val_hat, mag_val, mag_val_hat = model.forward(x_val_cuda, knobs_val_cuda)
            loss_val = loss_functions.calc_loss(y_val_hat, y_val_cuda, mag_val_hat)
            vl_avg = beta*vl_avg + (1-beta)*loss_val.item()    # (running) average val loss
            timediff = time.time() - first_time
            print(f"\repoch {epoch+1}/{epochs}, time: {timediff:.2f}: lr = {lr:.2e}, data_point {data_point}: loss: {smoothed_loss:.3e} val_loss: {vl_avg:.3e}   ",end="")


        #  Various forms of output...
        with open(logfilename, "a") as myfile:  # save progress of val loss to text file
            myfile.write(f"{epoch+1} {vl_avg:.3e}\n")

        if (epoch+1) % plot_every == 0:  # plot sample val data
            print("\nSaving sample data plots",end="")
            io_methods.plot_valdata(x_val_cuda, knobs_val_cuda, y_val_cuda, y_val_hat, effect, epoch, loss_val, target_size=y_size)

        # save checkpoint of model to file
        if ((epoch+1) % 25 == 0):
            print(f'\nsaving model to {checkpointname}',end="")
            state = {'epoch': epoch + 1, 'state_dict':  model.module.state_dict(),#model.state_dict(),
                'optimizer': optimizer.state_dict()}
            torch.save(state, checkpointname)

        if ((epoch+1) % 50 == 0) or (epoch == epochs-1):
            # Show magnitude data
            plt.figure(1)
            plt.imshow(mag_val.data.cpu().numpy()[0, :, :].T, aspect='auto', origin='lower')
            plt.title('Initial magnitude')
            io_methods.savefig('mag.png')
            plt.figure(2)  # <---- Check this out! Some "sub-harmonic" content is generated for the compressor if the analysis weights make only small perturbations
            plt.imshow(mag_val_hat.data.cpu().numpy()[0, :, :].T, aspect='auto', origin='lower')
            plt.title('Processed magnitude')
            io_methods.savefig('mag_hat.png')


            if isinstance(model, nn_proc.AsymMPAEC):     # Plot the spectrograms
                plt.matshow(model.dft_analysis.conv_analysis_real.weight.data.cpu().numpy()[:, 0, :] + 1)
                plt.title('Conv-Analysis Real')
                io_methods.savefig('conv_anal_real.png')
                plt.matshow(model.dft_analysis.conv_analysis_imag.weight.data.cpu().numpy()[:, 0, :])
                plt.title('Conv-Analysis Imag')
                io_methods.savefig('conv_anal_imag.png')
                plt.matshow(model.dft_synthesis.conv_synthesis_real.weight.data.cpu().numpy()[:, 0, :])
                plt.title('Conv-Synthesis Real')
                io_methods.savefig('conv_synth_real.png')
                plt.matshow(model.dft_synthesis.conv_synthesis_imag.weight.data.cpu().numpy()[:, 0, :])
                plt.title('Conv-Synthesis Imag')
                io_methods.savefig('conv_synth_imag.png')

            #io_methods.makemovie(datagen, model, batch_size)  #make a movie
    timediff = time.time() - first_time
    print("\ntimediff =",timediff)
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
    main(epochs=1000, n_data_points=200000, batch_size=200, device=device)#, effect=audio.Compressor())
# EOF

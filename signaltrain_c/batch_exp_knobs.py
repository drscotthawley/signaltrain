#/usr/bin/env python3.6
# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import numpy as np
import torch
from nn_modules import nn_proc
from helpers import audio, io_methods, learningrate
from losses import loss_functions
from subprocess import call, PIPE
import torch.nn as nn
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

import time
#from multiprocessing import Pool
#from functools import partial

def main(epochs=100, n_data_points=1, batch_size=20, device=torch.device("cuda:0"), change_every=5, effect=audio.Compressor_4c()):
    '''
       change_every = 5  # experiments suggest 5 to 20  here.  Lower=more accurate but slower, Higher=faster but higher loss
    '''

    # Data settings
    scale_factor = 0.5  # change dimensionality of run by this factor
    time_series_length = int(8192 * scale_factor)
    sampling_freq = 44100.
    # Analysis parameters
    ft_size = int(1024 * scale_factor)
    hop_size = int(384 * scale_factor)
    expected_time_frames = int(np.ceil(time_series_length/float(hop_size)) + np.ceil(ft_size/float(hop_size)))

    # print info about this run
    print(f'SignalTrain execution began at {time.time()}. Options:')
    print(f'    epochs = {epochs}, n_data_points = {n_data_points}, batch_size = {batch_size}, change_every = {change_every}')
    print(f'    scale_factor = {scale_factor}:  time_series_length = {time_series_length}, ft_size = {ft_size}, hop_size = {hop_size}')
    effect.info()  # Print effect settings

    # Initialize nn modules
    model = nn_proc.MPAEC(expected_time_frames, ft_size=ft_size, hop_size=hop_size, n_knobs=len(effect.knob_names))
    #model = nn_proc.Ensemble(model, N=4)

    # load from checkpoint
    checkpointname = 'modelcheckpoint.tar'
    if os.path.isfile(checkpointname):
        print("Checkpoint file found. Loading weights.")
        checkpoint = torch.load(checkpointname,) # map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.device_count() > 1:       # For Hawley's 2 GPUs this cuts execution time down by ~30% (not 50%)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)


    # learning rate schedule...although we don't bother stepping the momentum
    lr_max = 5e-4                         # obtained after running helpers.learningrate.lr_finder.py
    lrs = learningrate.get_1cycle_schedule(lr_max=lr_max, n_data_points=n_data_points, epochs=epochs, batch_size=batch_size)

    # Initialize optimizer. given our "random" training data, weight decay doesn't help but rather slows training way down
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lrs[0], weight_decay=0)

    # setup log file
    logfilename = "vl_avg_out.dat"
    with open(logfilename, "w") as myfile:  # save progress of val loss
        myfile.close()

    # set up data
    datagen = audio.AudioDataGenerator(time_series_length, sampling_freq, effect, batch_size=batch_size, device=device)#, path="Train/") # read from file takes a bit longer
    datagen_val = audio.AudioDataGenerator(time_series_length, sampling_freq, effect, batch_size=batch_size, requires_grad=False, device=device)#, path="Val/")

    x_cuda, y_cuda, knobs_cuda = datagen.new()
    x_val_cuda, y_val_cuda, knobs_val_cuda = datagen_val.new()


    # epochs
    vl_avg, beta = 0.0, 0.98  # for reporting, average over last 50 iterations
    iter_count = 0
    first_time = time.time()
    for epoch in range(epochs):
        print("")
        avg_loss, batch_num  = 0, 0
        data_point=0
        for batch in range(n_data_points//batch_size):

                lr = lrs[min(iter_count, len(lrs)-1)]
                data_point += batch_size

                # get new data every so many BATCHES (i.e. so many iterations)
                if (batch % change_every == 0): x_cuda, y_cuda, knobs_cuda = datagen.new()

                # forward synthesis
                x_hat, mag, mag_hat = model.forward(x_cuda, knobs_cuda)
                loss = loss_functions.calc_loss(x_hat, y_cuda, mag_hat)

                # validation data
                if False and (epoch % 5 == 0):  # adjust val data every this many EPOCHS
                    x_val_cuda, y_val_cuda, knobs_val_cuda =  datagen_val.new()#recyc_x=True)
                x_val_hat, mag_val, mag_val_hat = model.forward(x_val_cuda, knobs_val_cuda)
                loss_val = loss_functions.calc_loss(x_val_hat, y_val_cuda, mag_val_hat)

                vl_avg = beta*vl_avg + (1-beta)*loss_val.item()    # (running) average val loss

                batch_num += 1
                avg_loss = beta * avg_loss + (1-beta) *loss.item()
                smoothed_loss = avg_loss / (1 - beta**batch_num)

                timediff = time.time() - first_time
                print(f"\repoch {epoch+1}/{epochs}: lr = {lr:.2e}, data_point {data_point}: smoothed_loss: {smoothed_loss:.3e} vl_avg: {vl_avg:.3e} time: {timediff:.2f}  ",end="")
                # Opt
                optimizer.zero_grad()
                loss.backward()

                if torch.cuda.device_count() > 1:
                    for child in model.children():  # for DataParallel
                        child.clip_grad_norm_()
                else:
                    model.clip_grad_norm_()   # for non-DataParallel

                optimizer.step()

                optimizer.param_groups[0]['lr'] = lr     # adjust according to schedule
                iter_count += 1

        #  Various forms of output...
        io_methods.plot_valdata(x_val_cuda, knobs_val_cuda, y_val_cuda, x_val_hat, effect, epoch, loss_val)
        with open(logfilename, "a") as myfile:  # save progress of val loss
            myfile.write(f"{epoch+1} {vl_avg:.3e}\n")

        # save model to file  (fast)
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


            if isinstance(model, nn_proc.MPAEC):     # Plot the dictionaries
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
    main(epochs=10000, n_data_points=8000, batch_size=200, device=device)#, effect=audio.LowPass())
# EOF

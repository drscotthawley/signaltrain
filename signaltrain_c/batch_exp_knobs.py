#/usr/bin/env python3.6
# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from nn_modules import cls_fe_dft, nn_proc
from helpers import audio, io_methods
from losses import loss_functions
from subprocess import call, PIPE
import torch.nn as nn
from multiprocessing import Pool
from functools import partial
import os

import torch.nn.functional as F

def calc_loss(x_hat,y_cuda,mag,objective,batch_size=20):
        # Reconstruction term plus regularization -> Slightly less wiggly waveform
        loss = objective(x_hat, y_cuda) + 1e-5*mag.norm(1) #- 1e-3*F.conv1d(x_hat.unsqueeze(0),y_cuda.unsqueeze(0)).norm(1)
        return loss/batch_size


class DataGenerator():
    def __init__(self, time_series_length, sampling_freq, effect, batch_size=10, requires_grad=True, device=torch.device("cuda:0")):
        super(DataGenerator, self).__init__()
        self.time_series_length = time_series_length
        self.t = np.arange(time_series_length,dtype=np.float32) / sampling_freq
        self.effect = effect
        self.batch_size = batch_size
        self.requires_grad = requires_grad

        # preallocate memory
        self.x = np.zeros((batch_size,time_series_length),dtype=np.float32)
        self.y = np.zeros((batch_size,time_series_length),dtype=np.float32)
        self.knobs = np.zeros((batch_size,len(self.effect.knob_ranges)),dtype=np.float32)

    def gen_single(self, chooser=None, knobs=None, recyc_x=None):
        """create a single time-series"""
        if chooser is None:
            chooser = np.random.choice([0,1,2,4,6,7])  # for compressor
            #chooser = np.random.choice([1,3,5,6,7])  # for echo

        if recyc_x is None:
            x = audio.synth_input_sample(self.t, chooser)
        else:
            x = recyc_x   # don't generate new x

        if knobs is None:
            knobs = audio.random_ends(len(self.effect.knob_ranges))-0.5  # inputs to NN, zero-mean...except we emphasize the ends slightly
        y, x = self.effect.go(x, knobs)

        return x, y, knobs

    def new(self,chooser=None, knobs=None, recyc_x=False):

        # was going to try parallel via multiprocessing but it's actually slower than serial
        #self.pool = Pool(processes=10)
        knobs = None #audio.random_ends(len(self.effect.knob_ranges))-0.5  # same knobs for whole batch
        for line in range(self.batch_size):
            if recyc_x:
                #self.x[line,:], self.y[line,:], self.knobs[line,:] = self.pool.apply_async(partial(self.gen_single, chooser, knobs, recyc_x=self.x[line,:])).get()
                self.x[line,:], self.y[line,:], self.knobs[line,:] = self.gen_single(chooser, knobs=knobs, recyc_x=self.x[line,:])
            else:
                #self.x[line,:], self.y[line,:], self.knobs[line,:] = self.pool.apply_async(partial(self.gen_single,chooser,knobs)).get()
                self.x[line,:], self.y[line,:], self.knobs[line,:] = self.gen_single(chooser, knobs=knobs)
        #pool.close()

        x_cuda = torch.autograd.Variable(torch.from_numpy(self.x).to(device), requires_grad=self.requires_grad).float()
        y_cuda = torch.autograd.Variable(torch.from_numpy(self.y).to(device), requires_grad=False).float()
        knobs_cuda =  torch.autograd.Variable(torch.from_numpy(self.knobs).to(device), requires_grad=self.requires_grad).float()
        return x_cuda, y_cuda, knobs_cuda

    def new_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.x = np.zeros((batch_size,self.time_series_length),dtype=np.float32)
        self.y = np.zeros((batch_size,self.time_series_length),dtype=np.float32)
        self.knobs = np.zeros((batch_size,len(self.effect.knob_ranges)),dtype=np.float32)


def savefig(*args, **kwargs):  # little helper to close figures
    plt.savefig(*args, **kwargs)
    plt.close(plt.gcf())       # without this you eventually get 'too many figures open' message


def plot_valdata(x_val_cuda, knobs_val_cuda, y_val_cuda, x_val_hat, effect, epoch, loss_val, filename='val_data.png'):
    plt.figure(7,figsize=(6,8))
    knobs_w = effect.knobs_wc( knobs_val_cuda.data.cpu().numpy()[0,:] )
    titlestr = f'{effect.name} val data, epoch {epoch}, loss_val = {loss_val.item():.3e}\n'
    for i in range(len(effect.knob_names)):
        titlestr += f'{effect.knob_names[i]} = {knobs_w[i]:.2f}'
        if i < len(effect.knob_names)-1: titlestr += ', '
    plt.suptitle(titlestr)
    plt.subplot(3, 1, 1)
    plt.plot(x_val_cuda.data.cpu().numpy()[0, :], 'b', label='Input')
    plt.ylim(-1,1)
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(y_val_cuda.data.cpu().numpy()[0, :], 'r', label='Target')
    plt.ylim(-1,1)
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(y_val_cuda.data.cpu().numpy()[0, :], 'r', label='Target')
    plt.plot(x_val_hat.data.cpu().numpy()[0, :], c=(0,0.5,0,0.8), label='Predicted')
    plt.ylim(-1,1)
    plt.legend()
    savefig(filename)


def main(epochs=100, n_data_points=1, batch_size=20, device=torch.device("cuda:0")):

    # Data settings
    shrink_factor = 2  # reduce dimensionality of run by this factor
    time_series_length = 8192 // shrink_factor
    sampling_freq = 44100. // shrink_factor

    # Effect settings
    effect = audio.Compressor_new()
    print("effect.knob_names = ",effect.knob_names)

    # Analysis parameters
    ft_size = 1024 // shrink_factor
    hop_size = 384 // shrink_factor
    expected_time_frames = int(np.ceil(time_series_length/float(hop_size)) + np.ceil(ft_size/float(hop_size)))

    # Initialize nn modules
    #model = nn_proc.Ensemble(expected_time_frames, ft_size=ft_size, hop_size=hop_size)
    model = nn_proc.MPAEC(expected_time_frames, ft_size=ft_size, hop_size=hop_size, n_knobs=len(effect.knob_names))

    # load from checkpoint
    checkpointname = 'modelcheckpoint.tar'
    if os.path.isfile(checkpointname):
        print("Checkpoint file found. Loading weights.")
        checkpoint = torch.load(checkpointname,) # map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

    model = nn.DataParallel(model, device_ids=[0, 1])

    # learning rate modification for 1cycle learning & cosine annealing, following fastai
    #  ...although we don't bother stepping the momentum
    lr_max = 5e-4  # obtained after running helpers.lr_finder
    pct_start = 0.3
    div_factor = 25.
    lr_start = lr_max/div_factor
    lr_end = lr_start/1e4
    n_iter = n_data_points * epochs // batch_size
    a1 = int(n_iter * pct_start)   # length of first phase of cycle (linear growth)
    a2 = n_iter - a1               # length of second phase (cosine annealing)
    lrs_first = np.linspace(lr_start, lr_max, a1)
    lrs_second = (lr_max-lr_end)*(1+np.cos(np.linspace(0,np.pi,a2)))/2 + lr_end
    lrs = np.concatenate((lrs_first, lrs_second))  # look-up table containing LR schedule

    # Initialize optimizer. given our "random" training data, weight decay doesn't help but rather slows training way down
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr_start, weight_decay=0)


    # objective for loss functions
    objective = loss_functions.logcosh #mae with curvy bottom

    # setup log file
    logfilename = "vl_avg_out.dat"
    with open(logfilename, "w") as myfile:  # save progress of val loss
        myfile.close()

    # set up data
    datagen = DataGenerator(time_series_length, sampling_freq, effect, batch_size=batch_size, device=device)
    datagen_val = DataGenerator(time_series_length, sampling_freq, effect, batch_size=batch_size, requires_grad=False, device=device)

    x_cuda, y_cuda, knobs_cuda = datagen.new()
    x_val_cuda, y_val_cuda, knobs_val_cuda = datagen_val.new()

    # epochs
    vl_avg = 0.0
    beta = 0.98  # for reporting, average over last 50 iterations
    iter_count = 0
    for epoch in range(epochs):
        print("")
        avg_loss, batch_num  = 0, 0
        data_point=0
        for batch in range(n_data_points//batch_size):
                lr = lrs[min(iter_count, len(lrs)-1)]
                data_point += batch_size

                # get new data every few times
                if (batch % 5 == 0): x_cuda, y_cuda, knobs_cuda = datagen.new()
                # forward synthesis
                x_hat, mag, mag_hat = model.forward(x_cuda, knobs_cuda)
                loss = calc_loss(x_hat, y_cuda, mag_hat, objective,batch_size=batch_size)

                # validation data
                if (epoch % 5 == 0):  # adjust val settings every...
                    x_val_cuda, y_val_cuda, knobs_val_cuda =  datagen_val.new()#recyc_x=True)
                x_val_hat, mag_val, mag_val_hat = model.forward(x_val_cuda, knobs_val_cuda)
                loss_val = calc_loss(x_val_hat,y_val_cuda,mag_val_hat,objective,batch_size=batch_size)
                vl_avg = beta*vl_avg + (1-beta)*loss_val.item()

                batch_num += 1
                avg_loss = beta * avg_loss + (1-beta) *loss.item()
                smoothed_loss = avg_loss / (1 - beta**batch_num)

                print(f"\repoch {epoch+1}/{epochs}: lr = {lr:.2e}, data_point {data_point}: smoothed_loss: {smoothed_loss:.3e} vl_avg: {vl_avg:.3e}   ",end="")
                # Opt
                optimizer.zero_grad()
                loss.backward()

                for child in model.children():
                    child.clip_grad_norm_()
                optimizer.step()

                optimizer.param_groups[0]['lr'] = lr # adjust according to schedule
                iter_count += 1

        #  Various forms of output
        plot_valdata(x_val_cuda, knobs_val_cuda, y_val_cuda, x_val_hat, effect, epoch, loss_val)
        with open(logfilename, "a") as myfile:  # save progress of val loss
            myfile.write(f"{epoch+1} {vl_avg:.3e}\n")

        # save model to file
        if ((epoch+1) % 10 == 0):
            print(f'saving model to {checkpointname}',end="")
            state = {'epoch': epoch + 1, 'state_dict':  model.module.state_dict(),#model.state_dict(),
                'optimizer': optimizer.state_dict()}
            torch.save(state, checkpointname)

        if ((epoch+1) % 50 == 0) or (epoch == epochs-1):
            # Show magnitude data
            plt.figure(1)
            plt.imshow(mag_val.data.cpu().numpy()[0, :, :].T, aspect='auto', origin='lower')
            plt.title('Initial magnitude')
            savefig('mag.png')
            plt.figure(2)  # <---- Check this out! Some "sub-harmonic" content is generated for the compressor if the analysis weights make only small perturbations
            plt.imshow(mag_val_hat.data.cpu().numpy()[0, :, :].T, aspect='auto', origin='lower')
            plt.title('Processed magnitude')
            savefig('mag_hat.png')


            if False:     # Plot the dictionaries
                plt.matshow(model.dft_analysis.conv_analysis_real.weight.data.cpu().numpy()[:, 0, :] + 1)
                plt.title('Conv-Analysis Real')
                savefig('conv_anal_real.png')
                plt.matshow(model.dft_analysis.conv_analysis_imag.weight.data.cpu().numpy()[:, 0, :])
                plt.title('Conv-Analysis Imag')
                savefig('conv_anal_imag.png')
                plt.matshow(model.dft_synthesis.conv_synthesis_real.weight.data.cpu().numpy()[:, 0, :])
                plt.title('Conv-Synthesis Real')
                savefig('conv_synth_real.png')
                plt.matshow(model.dft_synthesis.conv_synthesis_imag.weight.data.cpu().numpy()[:, 0, :])
                plt.title('Conv-Synthesis Imag')
                savefig('conv_synth_imag.png')

            #io_methods.makemovie(datagen, model, batch_size)  #make a movie

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

    main(epochs=10000, n_data_points=8000, batch_size=200, device=device)
    print("")
# EOF

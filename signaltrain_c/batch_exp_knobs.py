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
from helpers import audio
from losses import loss_functions
from subprocess import call, PIPE
import torch.nn as nn


def calc_loss(x_hat,y_cuda,mag,objective,batch_size=20):
        # Reconstruction term plus regularization -> Slightly less wiggly waveform
        loss = objective(x_hat, y_cuda) + 1e-5*mag.norm(1)#/mag.nelement()
        return loss/batch_size


class DataGenerator():
    def __init__(self, time_series_length, sampling_freq, knob_ranges, batch_size=10, requires_grad=True, device=torch.device("cuda:0")):
        super(DataGenerator, self).__init__()
        self.time_series_length = time_series_length
        self.t = np.arange(time_series_length,dtype=np.float32) / sampling_freq
        self.knob_ranges = knob_ranges   # middle value of knobs
        self.batch_size = batch_size
        self.requires_grad = requires_grad

        # preallocate memory
        self.x = np.zeros((batch_size,time_series_length),dtype=np.float32)
        self.y = np.zeros((batch_size,time_series_length),dtype=np.float32)
        self.knobs = np.zeros((batch_size,len(knob_ranges)),dtype=np.float32)

    def gen_single(self, chooser=None, knobs=None, recyc_x=None):
        "create a single time-series"
        if chooser is None:
            chooser = np.random.choice([0,1,2,4,6,7])

        if recyc_x is None:
            x = audio.synth_input_sample(self.t, chooser)
        else:
            x = recyc_x   # don't generate new x

        if knobs is None:
            knobs = np.random.rand(3)-0.5  # inputs to NN, zero-mean
        knobs_w = self.knob_ranges[:,0] + (knobs+0.5)*(self.knob_ranges[:,1]-self.knob_ranges[:,0])
        y = audio.compressor(x=x, thresh=knobs_w[0], ratio=knobs_w[1], attack=knobs_w[2])

        return x, y, knobs

    def new(self,chooser=None, knobs=None, recyc_x=False):

        # was going to try parallel via multiprocessing but it's actually slower than serial
        #pool = Pool(processes=4)
        for line in range(self.batch_size):
            if recyc_x:
            #    result = pool.apply_async(partial(self.gen_single,chooser,knobs, recyc_x_cpu[line,:])).get()
                self.x[line,:], self.y[line,:], self.knobs[line,:] = self.gen_single(chooser,knobs, recyc_x=self.x[line,:])

            else:
            #    result = pool.apply_async(partial(self.gen_single,chooser,knobs)).get()
            #self.x[line,:], self.y[line,:], self.knobs[line,:] = result[0], result[1], result[2]
                self.x[line,:], self.y[line,:], self.knobs[line,:] = self.gen_single(chooser,knobs)
        #pool.close()

        x_cuda = torch.autograd.Variable(torch.from_numpy(self.x).to(device), requires_grad=self.requires_grad).float()
        y_cuda = torch.autograd.Variable(torch.from_numpy(self.y).to(device), requires_grad=False).float()
        knobs_cuda =  torch.autograd.Variable(torch.from_numpy(self.knobs).to(device), requires_grad=self.requires_grad).float()
        return x_cuda, y_cuda, knobs_cuda

    def new_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.x = np.zeros((batch_size,self.time_series_length),dtype=np.float32)
        self.y = np.zeros((batch_size,self.time_series_length),dtype=np.float32)
        self.knobs = np.zeros((batch_size,len(self.knob_ranges)),dtype=np.float32)


def savefig(*args, **kwargs):  # little helper to close figures
    plt.savefig(*args, **kwargs)
    plt.close(plt.gcf())       # without this you eventually get 'too many figures open' message


def plot_valdata(x_val_cuda, knobs_val_cuda, y_val_cuda, x_val_hat, knob_ranges, epoch, loss_val, filename='val_data.png'):
    plt.figure(7)
    plt.clf();
    knobs_w = knob_ranges[:,0]+ (knobs_val_cuda.data.cpu().numpy()+0.5) * (knob_ranges[:,1]-knob_ranges[:,0])  # world values
    threshold, ratio, attack = knobs_w[0,0], knobs_w[0,1],knobs_w[0,2]
    plt.title(f'Validation Data, epoch {epoch}, loss_val = {loss_val.item():.2f}'+ \
        f'\nthreshold = {threshold:.2f}, ratio = {ratio:.2f}, attack = {attack:.2f}' )
    plt.plot(x_val_cuda.data.cpu().numpy()[0, :], 'b', label='Input')
    plt.plot(y_val_cuda.data.cpu().numpy()[0, :], 'r', label='Target')
    plt.plot(x_val_hat.data.cpu().numpy()[0, :], c=(0,0.5,0,0.75), label='Predicted')
    plt.ylim(-1,1)
    plt.legend()
    savefig(filename)


def main_compressor(epochs=100, n_data_points=1, batch_size=20, device=torch.device("cuda:0")):

    # Data settings
    shrink_factor = 2  # reduce dimensionality of run by this factor
    time_series_length = 8192 // shrink_factor
    sampling_freq = 44100. // shrink_factor

    # Compressor settings
    '''threshold = -14
    ratio = 3
    attack = 4096 // shrink_factor
    knob_means = np.array([threshold, ratio, attack])'''
    knob_ranges = np.array([[-30,0], [1,5], [10,2048]]) # threshold, ratio, attack_release

    #attack = 0.3 / shrink_factor
    # Analysis parameters
    ft_size = 1024 // shrink_factor
    hop_size = 384 // shrink_factor
    expected_time_frames = int(np.ceil(time_series_length/float(hop_size)) + np.ceil(ft_size/float(hop_size)))

    # Initialize nn modules
    model = nn_proc.MPAEC(expected_time_frames, ft_size=ft_size, hop_size=hop_size)
    model.to(device)

    # learning rate modification for sgd & 1cycle learning
    n_batches = n_data_points // batch_size
    lr_min, lr_max = 2e-7, 1e-3
    lr_start = 1e-6
    lrsched_epochs = epochs//2.2    # timescale for sections of learning cycle
    lrsched_mult = 0.5
    lr_slope = (lr_max - lr_start)/(lrsched_epochs)

    # Initialize optimizer
    lr = lr_start
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)


    # objective for loss functions
    objective = loss_functions.logcosh #mae

    # setup log file
    logfilename = "vl_avg_out.dat"
    with open(logfilename, "w") as myfile:  # save progress of val loss
        myfile.close()

    # set up data
    datagen = DataGenerator(time_series_length, sampling_freq, knob_ranges, batch_size=batch_size, device=device)
    datagen_val = DataGenerator(time_series_length, sampling_freq, knob_ranges, batch_size=batch_size, requires_grad=False, device=device)
    datagen_movie = DataGenerator(time_series_length, sampling_freq, knob_ranges, batch_size=batch_size, requires_grad=False, device=device)

    x_cuda, y_cuda, knobs_cuda = datagen.new()
    x_val_cuda, y_val_cuda, knobs_val_cuda = datagen_val.new()

    vl_avg = 0.0
    for epoch in range(epochs):
        print("")
        avg_loss, batch_num  = 0, 0
        beta = 0.98  # for reporting, average over last 50 iterations
        data_point=0
        for batch in range(n_data_points//batch_size):

                data_point += batch_size  # stupid hack; fix later
                # get new data
                x_cuda, y_cuda, knobs_cuda = datagen.new()
                # forward synthesis
                x_hat, mag, mag_hat = model.forward(x_cuda, knobs_cuda)
                loss = calc_loss(x_hat,y_cuda,mag,objective,batch_size=batch_size)
                # validation data
                if False and (epoch % 5 == 0): # turn off for now
                    x_val_cuda, y_val_cuda, knobs_val_cuda = datagen_val.new()
                else:
                    x_val_cuda, y_val_cuda, knobs_val_cuda =  datagen_val.new(recyc_x=True)
                x_val_hat, mag_val, mag_val_hat = model.forward(x_val_cuda, knobs_val_cuda)
                loss_val = calc_loss(x_val_hat,y_val_cuda,mag_val,objective,batch_size=batch_size)
                vl_avg = beta*vl_avg + (1-beta)*loss_val.item()

                batch_num += 1
                avg_loss = beta * avg_loss + (1-beta) *loss.item()
                smoothed_loss = avg_loss / (1 - beta**batch_num)

                print(f"\repoch {epoch+1}/{epochs}: lr = {lr:.2e}, data_point {data_point}: smoothed_loss: {smoothed_loss:.3e} vl_avg: {vl_avg:.3e}     ",end="")
                # Opt
                optimizer.zero_grad()
                loss.backward()
                model.clip_grad_norm_()
                optimizer.step()

        #  Various forms of output
        plot_valdata(x_val_cuda, knobs_cuda, y_val_cuda, x_val_hat, knob_ranges, epoch, loss_val)
        with open(logfilename, "a") as myfile:  # save progress of val loss
            myfile.write(f"{epoch+1} {vl_avg:.3e}\n")

        # save model to file
        if ((epoch+1) % 10 == 0):
            checkpointname = 'modelcheckpoint.tar'
            print(f'saving model to {checkpointname}',end="")
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}
            torch.save(state, checkpointname)

        # learning rate schedule
        #if ((epoch+1) % lrsched_epochs == 0):
        #    lr = max( min_lr, lr*lrsched_mult)
        # 1cycle policy  @sgugger, https://sgugger.github.io/the-1cycle-policy.html
        if ((epoch) == lrsched_epochs):  # after one half-cycle, make it go back down
            lr_slope *= -1
        if ((epoch) % (2*lrsched_epochs) == 0) and ((epoch) >= 2*lrsched_epochs): # after one full cycle, shrink it every half cycle (and keep going down)
            lr *= 0.8
            lr_slope = -0.01*lr
            #n_data_points = min(2*n_data_points, 4000*64)
            #batch_size = min(2*batch_size, 20*64)
            #datagen.new_batch_size(batch_size)

        lr = max(lr_min, lr + lr_slope)
        optimizer.param_groups[0]['lr'] = lr


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

            # Plot the dictionaries
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
            #plt.show(block=True)
            '''
            print("\n\nMaking movies")
            for sig_type in [0,4]:
                print()
                x_val_cuda2, y_val_cuda2, knobs_val_cuda2 = datagen_movie.new(chooser=sig_type)
                frame = 0
                intervals = 7
                for t in np.linspace(-0.5,0.5,intervals):          # threshold
                    for r in np.linspace(-0.5,0.5,intervals):      # ratio
                        for a in np.linspace(-0.5,0.5,intervals):  # attack
                            frame += 1
                            print(f'\rframe = {frame}/{intervals**3-1}.   ',end="")
                            knobs = np.array([t, r, a])
                            x_val_cuda2, y_val_cuda2, knobs_val_cuda2 = datagen_movie.new(knobs=knobs, recyc_x=True, chooser=sig_type)
                            x_val_hat2, mag_val2, mag_val_hat2 = model.forward(x_val_cuda2, knobs_val_cuda2)
                            loss_val2 = calc_loss(x_val_hat2,y_val_cuda2,mag_val2,objective,batch_size=batch_size)

                            framename = f'movie{sig_type}_{frame:04}.png'
                            print(f'Saving {framename}           ',end="")
                            plot_valdata(x_val_cuda2, knobs_val_cuda2, y_val_cuda2, x_val_hat2, knob_ranges, epoch, loss_val, filename=framename)
                shellcmd = f'rm -f movie{sig_type}.mp4; ffmpeg -framerate 10 -i movie{sig_type}_%04d.png -c:v libx264 -vf format=yuv420p movie{sig_type}.mp4; rm -f movie{sig_type}_*.png'
                p = call(shellcmd, stdout=PIPE, shell=True)

            x_val_cuda2, y_val_cuda2, knobs_val_cuda2 = x_val_cuda, y_val_cuda, knobs_val_cuda
            '''
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

    main_compressor(epochs=10000, n_data_points=4000, batch_size=20, device=device)

# EOF

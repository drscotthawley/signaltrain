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

from multiprocessing import Pool
from functools import partial

class DataGenerator():
    def __init__(self, time_series_length, sampling_freq, knob_means, batch_size=10, requires_grad=True):
        super(DataGenerator, self).__init__()
        self.time_series_length = time_series_length
        self.t = np.arange(time_series_length,dtype=np.float32) / sampling_freq
        self.knob_means = knob_means   # middle value of knobs
        self.batch_size = batch_size
        self.requires_grad = requires_grad

        # preallocate memory
        self.x = np.zeros((batch_size,time_series_length),dtype=np.float32)
        self.y = np.zeros((batch_size,time_series_length),dtype=np.float32)
        self.knobs = np.zeros((batch_size,len(knob_means)),dtype=np.float32)

    def gen_single(self,chooser,knobs,recyc_x=None):
        "create a single time-series"
        if chooser is None:
            chooser = np.random.choice([0,4])

        if recyc_x is None:
            x = audio.synth_input_sample(self.t, chooser)
        else:
            x = recyc_x   # don't generate new x

        if knobs is None:
            knobs = np.random.rand(3)-0.5  # inputs to NN, zero-mean
        knobs_w = self.knob_means * (1+knobs)  # world values
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

        x_cuda = torch.autograd.Variable(torch.from_numpy(self.x).cuda(), requires_grad=self.requires_grad).float()
        y_cuda = torch.autograd.Variable(torch.from_numpy(self.y).cuda(), requires_grad=False).float()
        knobs_cuda =  torch.autograd.Variable(torch.from_numpy(self.knobs).cuda(), requires_grad=self.requires_grad).float()
        return x_cuda, y_cuda, knobs_cuda


def fwd(x_cuda, y_cuda, knobs_cuda, args):
    (dft_analysis, aenc, phs_aenc, dft_synthesis, objective) = args
    # Forward analysis pass
    x_real, x_imag = dft_analysis.forward(x_cuda)

    # Magnitude-Phase computation
    #mag1 = torch.sqrt( x_real**2 + x_imag**2 )
    mag = torch.norm(torch.cat((x_real.unsqueeze(0), x_imag.unsqueeze(0)), 0), 2, dim=0)#.unsqueeze(0)
    #print(" mag1 - mag2  = ",mag1.data.cpu().numpy()[0,:,0] - mag.data.cpu().numpy()[0,:,0])
    phs = torch.atan2(x_imag, x_real+1e-6)

    # Processes Magnitude and phase individually
    mag_hat = aenc.forward(mag, knobs_cuda, skip_connections='sf')
    phs_hat = phs_aenc.forward(phs, knobs_cuda, skip_connections=False) + phs # <-- Slightly smoother convergence

    # Back to Real and Imaginary
    an_real = mag_hat * torch.cos(phs_hat)
    an_imag = mag_hat * torch.sin(phs_hat)

    # Forward synthesis pass
    x_hat = dft_synthesis.forward(an_real, an_imag)

    # Reconstruction term plus regularization -> Slightly less wiggly waveform
    loss = objective(x_hat, y_cuda) + 1e-2*mag.norm(1)

    return x_hat, mag, mag_hat, loss


def savefig(*args, **kwargs):  # little helper to close figures
    plt.savefig(*args, **kwargs)
    plt.close(plt.gcf())       # without this you eventually get 'too many figures open' message


def plot_valdata(x_val_cuda, knobs_val_cuda, y_val_cuda, x_val_hat, knob_means, epoch, loss_val, filename='val_data.png'):
    plt.figure(7)
    plt.clf();
    knobs_w = (knobs_val_cuda.data.cpu().numpy()+1) * knob_means  # world values
    threshold, ratio, attack = knobs_w[0,0], knobs_w[0,1],knobs_w[0,2]
    plt.title(f'Validation Data, epoch {epoch}, loss_val = {loss_val.item():.2f}'+ \
        f'\nthreshold = {threshold:.2f}, ratio = {ratio:.2f}, attack = {attack:.2f}' )
    plt.plot(x_val_cuda.data.cpu().numpy()[0, :], 'b', label='Original')
    plt.plot(y_val_cuda.data.cpu().numpy()[0, :], 'r', label='Target')
    plt.plot(x_val_hat.data.cpu().numpy()[0, :], 'g', label='Predicted')
    plt.ylim(-1,1)
    plt.legend()
    savefig('val_data.png')



def main_compressor(epochs=100, n_data_points=1, batch_size=20):
    # Data settings
    shrink_factor = 2  # reduce dimensionality of run by this factor
    time_series_length = 8192 // shrink_factor
    sampling_freq = 44100. // shrink_factor

    # Compressor settings
    threshold = -14
    ratio = 3
    attack = 4096 // shrink_factor
    knob_means = np.array([threshold, ratio, attack])
    knobs = knob_means*0

    #attack = 0.3 / shrink_factor
    # Analysis parameters
    ft_size = 1024 // shrink_factor
    hop_size = 384 // shrink_factor
    expected_time_frames = int(np.ceil(time_series_length/float(hop_size)) + np.ceil(ft_size/float(hop_size)))
    decomposition_rank = 25
    # Initialize nn modules
    # Front-ends
    dft_analysis = cls_fe_dft.Analysis(ft_size=ft_size, hop_size=hop_size)
    dft_synthesis = cls_fe_dft.Synthesis(ft_size=ft_size, hop_size=hop_size)

    # Latent processors
    aenc = nn_proc.AutoEncoder(expected_time_frames, decomposition_rank, len(knob_means))
    phs_aenc = nn_proc.AutoEncoder(expected_time_frames, 2, len(knob_means))

    # learning rate modification for sgd & 1cycle learning
    n_batches = n_data_points // batch_size
    min_lr, max_lr = 1e-5, 1e-3
    min_mom, max_mom = 0.85, 0.95
    lr, momentum = min_lr, max_mom
    # Initialize optimizer
    lr = max_lr
    optimizer = torch.optim.Adam(list(dft_analysis.parameters()) +
                                 list(dft_synthesis.parameters()) +
                                 list(aenc.parameters()) +
                                 list(phs_aenc.parameters()),
                                 lr=lr
                                 #,momentum=momentum
                                 #,weight_decay=1e-3
                                 )
    # Initialize a loss functional
    objective = loss_functions.logcosh

    # collect these references to pass elsewhere
    args = (dft_analysis, aenc, phs_aenc, dft_synthesis, objective)

    # set up data
    datagen = DataGenerator(time_series_length, sampling_freq, knob_means, batch_size=batch_size)
    datagen_val = DataGenerator(time_series_length, sampling_freq, knob_means, batch_size=batch_size, requires_grad=False)
    datagen_movie = DataGenerator(time_series_length, sampling_freq, knob_means, batch_size=batch_size, requires_grad=False)

    x_cuda, y_cuda, knobs_cuda = datagen.new()
    x_val_cuda, y_val_cuda, knobs_val_cuda = datagen_val.new()

    vl_avg = 0.0
    stepsize = n_data_points//batch_size * 50  # 10 epochs
    for epoch in range(epochs):
        print("")
        avg_loss, batch_num  = 0, 0
        beta = 0.98
        data_point=0
        for batch in range(n_data_points//batch_size):

                data_point += batch_size  # stupid hack; fix later
                # get new data
                x_cuda, y_cuda, knobs_cuda = datagen.new()


                # forward synthesis
                x_hat, mag, mag_hat, loss = fwd(x_cuda, y_cuda, knobs_cuda, args)

                # validation data
                if False and (epoch % 5 == 0): # turn off for now
                    x_val_cuda, y_val_cuda, knobs_val_cuda = datagen_val.new()
                else:
                    x_val_cuda, y_val_cuda, knobs_val_cuda =  datagen_val.new(recyc_x=True)
                x_val_hat, mag_val, mag_val_hat, loss_val = fwd(x_val_cuda, y_val_cuda, knobs_val_cuda, args)
                vl_avg = beta*vl_avg + (1-beta)*loss_val.item()

                batch_num += 1
                avg_loss = beta * avg_loss + (1-beta) *loss.item()
                smoothed_loss = avg_loss / (1 - beta**batch_num)

                print(f"\repoch {epoch}/{epochs}: lr = {lr:.2e}, data_point {data_point}: smoothed_loss: {smoothed_loss:.3e} vl_avg: {vl_avg:.3e}     ",end="")

                # Opt
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(list(dft_analysis.parameters()) +
                                              list(dft_synthesis.parameters()),
                                              max_norm=1., norm_type=1)
                optimizer.step()



        if ((epoch+1) % 1 == 0):
            if ((epoch+1) % 50 == 0):
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
                plt.matshow(dft_analysis.conv_analysis_real.weight.data.cpu().numpy()[:, 0, :] + 1)
                plt.title('Conv-Analysis Real')
                savefig('conv_anal_real.png')
                plt.matshow(dft_analysis.conv_analysis_imag.weight.data.cpu().numpy()[:, 0, :])
                plt.title('Conv-Analysis Imag')
                savefig('conv_anal_imag.png')
                plt.matshow(dft_synthesis.conv_synthesis_real.weight.data.cpu().numpy()[:, 0, :])
                plt.title('Conv-Synthesis Real')
                savefig('conv_synth_real.png')
                plt.matshow(dft_synthesis.conv_synthesis_imag.weight.data.cpu().numpy()[:, 0, :])
                plt.title('Conv-Synthesis Imag')
                savefig('conv_synth_imag.png')
                #plt.show(block=True)

            # Numpy conversion and plotting
            plot_valdata(x_val_cuda, knobs_cuda, y_val_cuda, x_val_hat, knob_means, epoch, loss_val)

        if (((epoch+1) % 50 == 0) or (epoch == epochs-1)):
            print("\n\nMaking movies",end="")
            # ffmpeg -framerate 10 -i movie0_%04d.png -c:v libx264 -vf format=yuv420p out0.mp4
            for sig_type in [0,4]:
                print()
                x_val_cuda2, y_val_cuda2, knobs_val_cuda2 = datagen_movie.new()
                frame = 0
                intervals = 3
                for t in np.linspace(-0.5,0.5,intervals):          # threshold
                    for r in np.linspace(-0.5,0.5,intervals):      # ratio
                        for a in np.linspace(-0.5,0.5,intervals):  # attack
                            frame += 1
                            print(f'\rframe = {frame}/{intervals**3-1}.   ',end="")
                            knobs = np.array([t, r, a])
                            x_val_cuda2, y_val_cuda2, knobs_val_cuda2 = datagen_movie.new(knobs=knobs, recyc_x=True, chooser=sig_type)
                            x_val_hat2, mag_val, mag_val_hat, loss_val = fwd(x_val_cuda2, y_val_cuda2, knobs_val_cuda2, args)
                            framename = f'movie{sig_type}_{frame:04}.png'
                            print(f'Saving {framename}',end="")
                            plot_valdata(x_val_cuda2, knobs_val_cuda2, y_val_cuda2, x_val_hat2, knob_means, epoch, loss_val, filename=framename)

    return None


if __name__ == "__main__":
    np.random.seed(218)
    torch.manual_seed(218)
    torch.cuda.manual_seed(218)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    main_compressor(epochs=1000, n_data_points=4000, batch_size=50)

# EOF

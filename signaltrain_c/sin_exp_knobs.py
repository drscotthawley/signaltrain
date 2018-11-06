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


def get_cuda_data(time_series_length, sampling_freq, knobs0, recyc_x=None, chooser=None, requires_grad=True, knobs=None):
    # Generate data
    if chooser is None:
        chooser = np.random.choice([0,4])
    if recyc_x is None:
        x = audio.synth_input_sample(np.arange(time_series_length) / sampling_freq, chooser)
    else:
        x = recyc_x.data.cpu().numpy()[0,:]
    #print("x.shape = ",x.shape)
    if knobs is None:
        knobs = audio.random_ends(3)-0.5  # inputs to NN, zero-mean
    [threshold, ratio, attack] = [ i[0]*(1+i[1]) for i in zip(knobs0,knobs) ] # convert to plugin control values
    y = audio.compressor(x=x, thresh=threshold, ratio=ratio, attack=attack)
    #y = audio.compressor_new(x=x, thresh=threshold, ratio=ratio, attackTime=attack, releaseTime=attack)

    # Reshape data
    x = x.reshape(1, time_series_length)
    y = y.reshape(1, time_series_length)

    x_cuda = torch.autograd.Variable(torch.from_numpy(x).cuda(), requires_grad=requires_grad).float()
    y_cuda = torch.autograd.Variable(torch.from_numpy(y).cuda(), requires_grad=False).float()
    knobs_cuda =  torch.autograd.Variable(torch.from_numpy(knobs).cuda(), requires_grad=requires_grad).float()
    return x_cuda, y_cuda, knobs_cuda


def fwd_analysis(x_cuda, y_cuda, knobs_cuda, args):
    (dft_analysis, aenc, phs_aenc, dft_synthesis, objective) = args
    # Forward analysis pass
    x_real, x_imag = dft_analysis.forward(x_cuda)

    # Magnitude-Phase computation
    mag = torch.norm(torch.cat((x_real, x_imag), 0), 2, dim=0).unsqueeze(0)
    phs = torch.atan2(x_imag, x_real+1e-6)

    # Processes Magnitude and phase individually
    mag_hat = aenc.forward(mag, knobs_cuda, skip_connections='sf')
    phs_hat = phs_aenc.forward(phs, knobs_cuda, skip_connections=False) + phs # <-- Slightly smoother convergence

    # Back to Real and Imaginary
    an_real = mag_hat * torch.cos(phs_hat)
    an_imag = mag_hat * torch.sin(phs_hat)

    # Forward synthesis pass
    x_hat = dft_synthesis.forward(an_real, an_imag)

    # skip connection
    x_hat = x_hat + x_cuda

    # Reconstruction term plus regularization -> Slightly less wiggly waveform
    loss = objective(x_hat, y_cuda) + 5e-3*mag.norm(1)

    return x_hat, mag, mag_hat, loss


def savefig(*args, **kwargs):  # little helper to close figures
    plt.savefig(*args, **kwargs)
    plt.close(plt.gcf())       # without this you eventually get 'too many figures open' message

def main_compressor(epochs=100, n_data_points=1, batch_size=20):
    # Data settings
    shrink_factor = 2  # reduce dimensionality of run by this factor
    time_series_length = 8192 // shrink_factor
    sampling_freq = 44100. // shrink_factor

    # Compressor settings
    threshold = -14
    ratio = 3
    attack = 4096 // shrink_factor
    knobs0 = [threshold, ratio, attack]

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
    aenc = nn_proc.AutoEncoder(expected_time_frames, decomposition_rank, len(knobs0))
    phs_aenc = nn_proc.AutoEncoder(expected_time_frames, 2, len(knobs0))

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
    x_cuda, y_cuda, knobs_cuda = get_cuda_data(time_series_length, sampling_freq, knobs0)
    x_val_cuda, y_val_cuda, knobs_val_cuda = get_cuda_data(time_series_length, sampling_freq, knobs0, requires_grad=False)

    vl_avg = 0.0
    iterations = 0
    stepsize = n_data_points//batch_size * 50  # 10 epochs
    for epoch in range(epochs):
        print("")
        loss, loss_val, loss_count, avg_loss, batch_num  = 0, 0, 0, 0, 0
        #batch_size = min( int(batch_size * 1.07), n_data_points/40)
        losses = []
        log_lrs = []
        beta = 0.98
        just_plot_once = False

        for data_point in range(n_data_points):
                # get new data
                x_cuda, y_cuda, knobs_cuda = get_cuda_data(time_series_length, sampling_freq, knobs0)

                # forward synthesis
                x_hat, mag, mag_hat, loss_1point = fwd_analysis(x_cuda, y_cuda, knobs_cuda, args)
                loss += loss_1point   # accumulate loss over mini-batch
                loss_count += 1

                if (data_point % batch_size == 0) or (data_point == n_data_points-1):
                    iterations += 1
                    if (epoch % 5 == 10): # turn off for now
                        x_val_cuda, y_val_cuda, knobs_val_cuda = get_cuda_data(time_series_length, sampling_freq, knobs0, requires_grad=False)
                    else:
                        x_val_cuda, y_val_cuda, knobs_val_cuda = get_cuda_data(time_series_length, sampling_freq, knobs0, recyc_x=x_val_cuda, requires_grad=False)
                    x_val_hat, mag_val, mag_val_hat, loss_val = fwd_analysis(x_val_cuda, y_val_cuda, knobs_val_cuda, args)
                    vl_avg = beta*vl_avg + (1-beta)*loss_val.item()

                    batch_num += 1
                    loss = loss / loss_count
                    avg_loss = beta * avg_loss + (1-beta) *loss.item()
                    smoothed_loss = avg_loss / (1 - beta**batch_num)

                    if (data_point % 20 == 0) or (data_point == n_data_points-1):
                        print(f"\repoch {epoch}/{epochs}: lr = {lr:.2e}, data_point {data_point}: loss: {loss:.3f} vl_avg: {vl_avg:.3f}     ",end="")

                    losses.append(smoothed_loss)
                    log_lrs.append(np.log10(lr))

                    # Opt
                    optimizer.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(list(dft_analysis.parameters()) +
                                                  list(dft_synthesis.parameters()),
                                                  max_norm=1., norm_type=1)
                    optimizer.step()

                    loss, loss_count = 0, 0


                    # learning rate scheduling (for SGD or Adam)
                    '''
                    measure_in = iterations
                    cycle = np.floor(1 + measure_in/(2*stepsize))
                    ramp = np.abs( measure_in / stepsize - 2*cycle + 1.0)
                    lr = max(min_lr, min_lr + (max_lr - min_lr)*max(0.0, 1.0-ramp))
                    momentum = max_mom - (max_mom - min_mom)*max(0.0, 1.0-ramp)
                    if (measure_in % (2*stepsize) == 0):
                        max_lr *= 0.3
                    if (lr >= min_lr):
                        optimizer.param_groups[0]['lr'] = lr
                    #optimizer.param_groups[0]['momentum'] = momentum
                    '''



        if ((epoch+1) % 1 == 0):
            plt.plot(log_lrs[10:-5],losses[10:-5])
            savefig('loss_vs_lr.png')
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
            plt.figure(7)
            plt.clf();
            x_val_hat_np = x_val_hat.data.cpu().numpy()
            x_val_cuda_np = x_val_cuda.data.cpu().numpy()
            y_val_cuda_np = y_val_cuda.data.cpu().numpy()
            [threshold, ratio, attack] = [ i[0]*(1+i[1]) for i in zip(knobs0, knobs_val_cuda.data.cpu().numpy()) ]
            plt.title(f'Validation Data, epoch {epoch}, loss_val = {loss_val.item():.2f}'+ \
                f'\nthreshold = {threshold:.2f}, ratio = {ratio:.2f}, attack = {attack:.2f}' )
            plt.plot(x_val_cuda_np[0, :], 'b', label='Original')
            plt.plot(y_val_cuda_np[0, :], 'r', label='Target')
            plt.plot(x_val_hat_np[0, :], 'g', label='Predicted')
            plt.ylim(-1,1)
            plt.legend()
            savefig('val_data.png')
            #plt.show(); plt.pause(0.001)
            if just_plot_once:
                print("\nsmoothed_loss = ",smoothed_loss,", aborting.")
                print("lr = ",lr)
                return

        if ((epoch+1) % 50 == 0) or (epoch == epochs-1):
            print("\n\nMaking movies")
            # ffmpeg -framerate 10 -i movie0_%04d.png -c:v libx264 -vf format=yuv420p out0.mp4
            for sig_type in [0,4]:
                x_val_cuda, y_val_cuda, knobs_val_cuda = get_cuda_data(time_series_length, sampling_freq, knobs0, requires_grad=False, chooser=sig_type)
                frame = 0
                intervals = 7
                for t in np.linspace(-0.5,0.5,intervals):          # threshold
                    for r in np.linspace(-0.5,0.5,intervals):      # ratio
                        for a in np.linspace(-0.5,0.5,intervals):  # attack
                            frame += 1
                            print(f'frame = {frame}/{intervals**3-1}.   ',end="")
                            knobs = np.array([t, r, a])
                            x_val_cuda, y_val_cuda, knobs_val_cuda = get_cuda_data(time_series_length, sampling_freq, knobs0, knobs=knobs, recyc_x=x_val_cuda, requires_grad=False, chooser=sig_type)
                            x_val_hat, mag_val, mag_val_hat, loss_val = fwd_analysis(x_val_cuda, y_val_cuda, knobs_val_cuda, args)

                            plt.figure(8)
                            plt.clf();
                            x_val_hat_np = x_val_hat.data.cpu().numpy()
                            x_val_cuda_np = x_val_cuda.data.cpu().numpy()
                            y_val_cuda_np = y_val_cuda.data.cpu().numpy()
                            [threshold, ratio, attack] = [ i[0]*(1+i[1]) for i in zip(knobs0, knobs_val_cuda.data.cpu().numpy()) ]
                            plt.title(f'\nthreshold = {threshold:.2f}, ratio = {ratio:.2f}, attack = {attack:.2f}' )
                            plt.plot(x_val_cuda_np[0, :], 'b', label='Original')
                            plt.plot(y_val_cuda_np[0, :], 'r', label='Target')
                            plt.plot(x_val_hat_np[0, :], 'g', label='Predicted')
                            plt.ylim(-1,1)
                            plt.legend()
                            framename = f'movie{sig_type}_{frame:04}.png'
                            print(f'Saving {framename}')
                            savefig(framename)

    return None


if __name__ == "__main__":
    np.random.seed(218)
    torch.manual_seed(218)
    torch.cuda.manual_seed(218)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    main_compressor(epochs=1000, n_data_points=4000)

# EOF

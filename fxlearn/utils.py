__author__ = 'S.H. Hawley'

# imports
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import shutil
import librosa
import os
from multiprocessing.pool import ThreadPool as Pool
from functools import partial


# my cheap way of showing loss 'graphically':  "*" farther right means larger loss (tilt your head to the right)
def print_loss(loss_val, log=True):
    if (log):
        term_width = 200
        logval = np.log10(loss_val)
        print(' loss = {:10.5e}'.format(loss_val) ,  ' '*int(term_width + 30*logval)+'*')
    else:
        term_width = 100
        print(' loss = {:10.5e}'.format(loss_val) ,  ' '*int(loss_val/1*term_width)+'*')
    return



def gen_pitch_shifted_pair(input_sigs, target_sigs, fs, amp_fac, freq_fac, num_waves, batch_index):
    sig_length = input_sigs.shape[1]
    for n in range(num_waves):
        # randomize the signal
        amp = 0.2*np.random.rand()    # stay bounded well below 1.0
        freq = 2 * np.pi * ( 400 + 400*np.random.rand() )

        # learn the adaptive filter for the following input -> target pair: different amp, freq & phase
        input_sigs[batch_index]  +=           amp * torch.cos(           freq * torch.arange(sig_length) / fs)
        target_sigs[batch_index] += amp_fac * amp * torch.sin(freq_fac * freq * torch.arange(sig_length) / fs)
    return


# currently creates pitch-shifted collections of sine waves
def make_signals(sig_length, fs=44100., amp_fac=0.43, freq_fac=0.35, num_waves=20, batch_size=20, parallel=True):

    input_sigs = torch.zeros((batch_size,sig_length))
    target_sigs = torch.zeros((batch_size,sig_length))

    # generate them in parallel threads that all share the input_sigs and target_sigs arrays
    batch_indices = tuple( range(batch_size) )
    if (parallel):
        pool = Pool()
        pool.map( partial(gen_pitch_shifted_pair, input_sigs, target_sigs, fs, amp_fac, freq_fac, num_waves), batch_indices)
    else:
        for batch_index in batch_indices:
            gen_pitch_shifted_pair(input_sigs, target_sigs, fs, amp_fac, freq_fac, num_waves, batch_index)

    input_var = Variable(input_sigs)
    target_var = Variable(target_sigs, requires_grad=False)
    if torch.has_cudnn:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, target_var



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



def make_report(input_var, target_var, wave_form, outfile=None, epoch=None, show_input=False):
    wave_form = wave_form.squeeze(1).data.cpu().numpy()[0, :]
    input_sig = input_var.squeeze(1).data.cpu().numpy()[0, :]
    target_sig = target_var.squeeze(1).data.cpu().numpy()[0, :]
    mse = np.mean((wave_form - target_sig)**2)
    print('MSE = ', mse)      # Print mean squared error

    fig = plt.figure(figsize=(11,8.5),dpi=120)
    if (show_input):
        plt.plot(input_sig, 'b-', label='Input')
    plt.plot(target_sig, 'r-', label='Target')
    plt.plot(wave_form, 'g-', label='Output')

    ampmax = 1.0
    plt.ylim((-ampmax,ampmax))   # zoom in
    plt.legend()

    outstr = 'MSE = '+str(mse)
    if (None != epoch):
        outstr += ', Epoch = '+str(epoch)
    plt.text(0, 0.8*ampmax, outstr )

    if (None == outfile):
        plt.show()
    else:
        plt.savefig(outfile)
        plt.close(fig)

    sr = 44100
    librosa.output.write_wav('progress_input.wav', input_sig, sr)
    librosa.output.write_wav('progress_output.wav', wave_form, sr)
    librosa.output.write_wav('progress_target.wav', target_sig, sr)

# EOF

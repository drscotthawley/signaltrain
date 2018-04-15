__author__ = 'S.H. Hawley'

# imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import shutil


# my cheap way of showing loss 'graphically':  "*" farther right means larger loss (tilt your head to the right)
def print_loss(loss_val):
    term_width = 100
    print(' loss = {:f}'.format(loss_val) ,  ' '*int(loss_val/1*term_width)+'*')
    return


# currently creates pitch-shifted collections of sine waves
def make_signals(sig_length, fs=44100., amp_fac=0.5, freq_fac = 0.35, num_waves = 20, batch_size=20):

    input_sig = torch.zeros((batch_size,sig_length))
    target_sig = torch.zeros((batch_size,sig_length))

    for batch in range(batch_size):
        for n in range(num_waves):
            # randomize the signal
            amp = np.random.rand()
            freq = 400 + 400*np.random.rand()

            # learn the adaptive filter for the following input -> target pair
            input_sig[batch] += amp * torch.cos(2 * np.pi * freq * torch.arange(sig_length) / fs)
            target_sig[batch] += amp*amp_fac * torch.cos(2 * np.pi * freq*freq_fac * torch.arange(sig_length) / fs)

    input_var = Variable(input_sig)
    target_var = Variable(target_sig, requires_grad=False)
    if torch.has_cudnn:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, target_var



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



def make_report(input_var, target_var, wave_form, outfile=None, epoch=None):
    wave_form = wave_form.squeeze(1).data.cpu().numpy()[0, :]
    input_sig = input_var.squeeze(1).data.cpu().numpy()[0, :]
    target_sig = target_var.squeeze(1).data.cpu().numpy()[0, :]
    mse = np.mean((wave_form - target_sig)**2)
    print('MSE = ', mse)      # Print mean squared error

    fig = plt.figure(figsize=(11,8.5),dpi=120)
    plt.plot(input_sig, 'b-', label='Input')
    plt.plot(target_sig, 'r-', label='Target')
    plt.plot(wave_form, 'g-', label='Output')
    plt.legend()

    outstr = 'MSE = '+str(mse)
    if (None != epoch):
        outstr += ', Epoch = '+str(epoch)
    plt.text(0, np.max(input_sig), outstr )

    if (None == outfile):
        plt.show()
    else:
        plt.savefig(outfile)
        plt.close(fig)


# EOF

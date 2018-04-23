__author__ = 'S.H. Hawley'

# imports
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import shutil
from . import audio as fxl_audio


# class for keeping track of and printing loss info
class LossLogger():
    def __init__(self):
        self.loss_hist = []
        self.epoch_hist = []
        self.vloss_hist = []

    def print_loss(self, loss_num, logy=True, vloss_num=None):
        if (logy):
            term_width = 200
            logval = np.log10(loss_num)
            print(' loss = {:10.5e}'.format(loss_num) ,  ' '*int(term_width + 30*logval)+'*', end="")
        else:
            term_width = 100
            print(' loss = {:10.5e}'.format(loss_num) ,  ' '*int(loss_num/1*term_width)+'*',end="")
        if (vloss_num is not None):
            print('  val = {:10.5e}'.format(vloss_num))

    def update(self, epoch, loss, vloss):
        loss_num = loss.data.cpu().numpy()[0]
        self.loss_hist.append([epoch, loss_num])
        if (vloss is not None):
            vloss_num = vloss.data.cpu().numpy()[0]
            self.vloss_hist.append([epoch, vloss_num])
        else:
            vloss_num = None
        self.print_loss(loss_num, vloss_num=vloss_num)


def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    # state can include multiple elements in dict format, e.g.
    #    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),'optimizer': optimizer.state_dict(),}
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
    return

def load_checkpoint(model, optimizer, losslogger, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        #best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger



def make_report(input_var, target_var, wave_form, loss_log, outfile=None, epoch=None, show_input=True,
    diff=False):
    mse =loss_log.loss_hist[-1]

    # convert to numpy arrays

    # --------------
    # make figure
    #--------------
    num_examples = 3
    fig, panels = plt.subplots(num_examples+2, figsize=(8.5,11),dpi=120 )

    # top panel: loss history
    lhist = np.array(loss_log.loss_hist)
    vlhist = np.array(loss_log.vloss_hist)
    panels[0].semilogy(lhist[:,0], lhist[:,1], label='Train')
    panels[0].semilogy(vlhist[:,0], vlhist[:,1], label='Val')
    panels[0].set_ylabel('Loss')
    panels[0].set_xlabel('Epoch')
    panels[0].legend()

    # middle panels: sample function(s)
    for example in range(num_examples):
        wf = wave_form.squeeze(1).data.cpu().numpy()[example, :]
        inp = input_var.squeeze(1).data.cpu().numpy()[example, :]
        tar = target_var.squeeze(1).data.cpu().numpy()[example, :]

        p = example+1
        if (show_input):
            panels[p].plot(inp, 'b-', label='Input')
        panels[p].plot(tar, 'r-', label='Target')
        panels[p].plot(wf, 'g-', label='Output')
        ampmax = 1.0 # 1.1*np.max(input_sig)
        panels[p].set_ylim((-ampmax,ampmax))   # zoom in
        outstr = 'MSE = '+str(mse)
        if (None != epoch):
            outstr += ', Epoch = '+str(epoch)
        panels[p].text(0, 0.85*ampmax, outstr )
        panels[p].legend()

    # panels[3]: difference
    diffval = wf - tar
    p = p+1
    panels[p].plot(diffval, 'k-', label='Output - Target')
    panels[p].set_ylabel('Output - Target')

    if (None == outfile):
        plt.show()
    else:
        plt.savefig(outfile+'.pdf')
        plt.close(fig)

    # save audio files
    sr = 44100
    fxl_audio.write_audio_file(outfile+'_input.wav', inp, sr)
    fxl_audio.write_audio_file(outfile+'_output.wav', wf, sr)
    fxl_audio.write_audio_file(outfile+'_target.wav', tar, sr)
    return

# EOF

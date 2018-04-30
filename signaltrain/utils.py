__author__ = 'S.H. Hawley'

# imports
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import shutil
from . import audio as st_audio


# class for keeping track of and printing loss info
class LossLogger():
    def __init__(self):
        self.loss_hist = []
        self.epoch_hist = []
        self.vloss_hist = []
        self.best_vloss = 999.9

    def print_loss(self, loss_num, logy=True, vloss_num=None):
        if (logy):
            term_width = 180
            logval = np.log10(loss_num)
            print(' loss: {:10.5e}'.format(loss_num) ,  ' '*int(term_width + 30*logval)+'*', end="")
        else:
            term_width = 100
            print(' loss: {:10.5e}'.format(loss_num) ,  ' '*int(loss_num/1*term_width)+'*',end="")
        if (vloss_num is not None):
            print('  val_loss: {:10.5e}'.format(vloss_num))

    def update(self, epoch, loss, vloss):
        loss_num = loss_num = loss.data.cpu().numpy()  # prior to pytorch 0.4, we needed a [0] on the end of this
        self.loss_hist.append([epoch, loss_num])
        if (vloss is not None):
            vloss_num = vloss.data.cpu().numpy() # [0]
            self.vloss_hist.append([epoch, vloss_num])
        else:
            vloss_num = None
        self.print_loss(loss_num, vloss_num=vloss_num)

    def is_best(self):   # only updates best_vloss when called; this is intentional
        vloss_num = (self.vloss_hist[-1])[-1]
        if (vloss_num < self.best_vloss):
            self.best_vloss = vloss_num
            return True
        return False



def save_checkpoint(state, filename='checkpoint.pth.tar', best_only=False, is_best=False):
    # state can include multiple elements in dict format, e.g.
    #    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),'optimizer': optimizer.state_dict(),}
    if is_best:
        print("Best val_loss so far (for checkpoint).    ", end="")

    if (not best_only) or is_best:
        print("Saving checkpoint in",filename)
        torch.save(state, filename)
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

# this is a simpler version of load_checkpoint that simply ignores everything but the model
def load_weights(model, filename='checkpoint.pth.tar'):
    if os.path.isfile(filename):
        print("=> Initializing model weights from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(filename))
    return model


def make_report(input_var, target_var, wave_form, loss_log, outfile=None, epoch=None, show_input=True,
    diff=False, mu_law=False):
    mse =loss_log.loss_hist[-1]


    # --------------
    # make figure
    #--------------
    num_signal_plots = 3
    fig, panels = plt.subplots(num_signal_plots+2, figsize=(8.5,11),dpi=120 )
    lw    =   1    # line width


    # top panel: loss history
    lhist = np.array(loss_log.loss_hist)
    vlhist = np.array(loss_log.vloss_hist)
    panels[0].loglog(lhist[:,0], lhist[:,1], label='Train', linewidth=lw)
    panels[0].loglog(vlhist[:,0], vlhist[:,1], label='Val', linewidth=lw)
    panels[0].set_ylabel('Loss')
    panels[0].set_xlabel('Epoch')
    panels[0].legend(loc=1)
    xmin = 1
    if (lhist[-1,0] > 100):  # ignore the 1st 100 epochs (better when plotting with loglog scale)
        xmin = 10
    panels[0].set_xlim(left=xmin)
    panels[0].set_ylim(top= vlhist[xmin,1] )


    if (input_var is None):  # exit
        plt.savefig(outfile+'.pdf')
        plt.close(fig)
        return

    # convert to numpy arrays
    i = np.random.randint(input_var.size()[0])  # pick a random example to plot
    inp = input_var.squeeze(1).data.cpu().numpy()[i, :]
    tar = target_var.squeeze(1).data.cpu().numpy()[i, :]
    wf = wave_form.squeeze(1).data.cpu().numpy()[i, :]

    if mu_law:
        inp = st_audio.decode_mu_law(inp)
        tar = st_audio.decode_mu_law(tar)
        wf  = st_audio.decode_mu_law(wf)

    # middle panels: sample function(s)
    for signal_plot in range(num_signal_plots):
        p = signal_plot+1
        if (0==signal_plot):
            panels[p].plot(inp, 'b-', label='Input', linewidth=lw)
        elif (signal_plot >=1):
            panels[p].plot(tar, 'r-', label='Target', linewidth=lw)
        if (2==signal_plot):
            panels[p].plot(wf, 'g-', label='Output', linewidth=lw)
        ampmax = 1.0 # 1.1*np.max(input_sig)
        panels[p].set_ylim((-ampmax,ampmax))   # zoom in
        if (0 == signal_plot):
            outstr = 'MSE = '+str(mse)
            if (None != epoch):
                outstr += ', Epoch = '+str(epoch)
            panels[p].text(0, 0.85*ampmax, outstr )
        panels[p].legend(loc=1)

    # panels[3]: difference
    diffval = wf - tar
    p = p+1
    panels[p].plot(diffval, 'k-', label='Output - Target', linewidth=lw)
    panels[p].set_ylabel('Output - Target')

    if (None == outfile):
        plt.show()
    else:
        plt.savefig(outfile+'.pdf')
        plt.close(fig)

    # save audio files
    sr = 44100
    st_audio.write_audio_file(outfile+'_input.wav', inp, sr)
    st_audio.write_audio_file(outfile+'_output.wav', wf, sr)
    st_audio.write_audio_file(outfile+'_target.wav', tar, sr)
    return

# EOF

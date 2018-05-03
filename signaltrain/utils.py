__author__ = 'S.H. Hawley'
__version__ = '0.0.1'

# imports
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import shutil
from . import audio as st_audio
import datetime  # just to name the log files


# simple progress bar for batch training. you can get a better ProgBar if you install something. ;-)
# ripped off the style of Keras
def progbar(epoch, max_epochs, batch_index, nbatches, loss, width=40, vloss=None):
    base_str = '\rEpoch '+str(epoch)+' /'+str(max_epochs)+': '
    percent = (batch_index+1)/nbatches
    barlength = int(width * percent)
    bar  = '='*barlength + '>'
    leftover = width - barlength
    space =  ' '*leftover
    loss_num = np.asarray(loss.data.cpu().numpy())
    print(base_str + bar + space+' loss: ' + '%.4f' % loss_num, end="")
    return


# class for keeping track of and printing loss info
class LossLogger():
    def __init__(self, name='run'):
        # name is the tag for the run. it will be appended with the current time & date
        #      a directory with this full name will be created, and logs will go there
        self.history = {'Epoch':[], 'Train':[], 'Val':[], 'Val_':[]}
        self.best_vloss = 999.9
        self.name = name + '_{date:%H:%M:%S_%b_%d_%Y}'.format( date=datetime.datetime.now() )
        self.lossfile = self.name + "/" + "losses.csv"
        self.setup_files()

    def setup_files(self):
        print('LossLogger: logging to directory "'+self.name+'"')
        os.mkdir(self.name)   # TODO: noclobber. Starting multiple simultaneous runs in the same directory
                              # with the same tag (unlikely) will fail here
        with open(self.lossfile, "a") as myfile:
            myfile.write('#Epoch, TrainLoss, ValTotalLoss, ValMSELoss, ValL1Loss\n')

    def update_file(self):  # append loss info to file
        with open(self.lossfile, "a") as myfile:
            outstr = str(self.history['Epoch'][-1])+', {:10.5e}'.format(self.history['Train'][-1])+ \
                ', {:10.5e}'.format(self.history['Val'][-1])
            for otherloss in self.history['Val_'][-1]:
                outstr += ', {:10.5e}'.format( otherloss )
            myfile.write(outstr+'\n')

    def update(self, epoch, loss, vloss, vlosses):
        tloss, vloss = loss.data.cpu().numpy(), vloss.data.cpu().numpy()
        #self.epoch_hist.append(epoch)
        self.history['Epoch'].append(epoch)
        self.history['Train'].append(tloss)
        self.history['Val'].append(vloss)
        vllist = []
        for otherloss in vlosses:
            vllist.append( otherloss.data.cpu().numpy() )
        self.history['Val_'].append(vllist)

        print('  val_loss: {:10.5e}'.format(self.history['Val'][-1]))
        self.update_file()

    def is_best(self):   # only updates best_vloss when called; this is intentional
        vloss_num = self.history['Val'][-1]  # most recent val loss
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


    # --------------
    # make figure
    #--------------
    num_signal_plots = 3
    fig, panels = plt.subplots(num_signal_plots+2, figsize=(8.5,11),dpi=120 )
    lw    =   1    # line width

    # top panel: loss history
    panels[0].set_title(loss_log.name+', Epoch '+str(loss_log.history['Epoch'][-1]))
    ehist = loss_log.history['Epoch']
    for key in ('Train','Val'):
        panels[0].loglog(ehist, loss_log.history[key], label=key, linewidth=lw)

    vlarr = np.array(loss_log.history['Val_'])
    for i in range(vlarr.shape[1]):   # the components of the validation loss
        panels[0].loglog(ehist, vlarr[:,i], label='Val['+str(i)+']', linewidth=lw)

    panels[0].set_ylabel('Loss')
    panels[0].set_xlabel('Epoch')
    panels[0].legend(loc=1)
    xmin = 1
    if (ehist[-1] > 100):  # ignore the 1st 10 epochs (better when plotting with loglog scale)
        xmin = 10
    panels[0].set_xlim(left=xmin)
    panels[0].set_ylim(top= loss_log.history['Val'][xmin] )


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
        panels[p].set_ylim((-ampmax,ampmax))   # for consistency
        panels[p].legend(loc=1)

    # panels[3]: difference
    diffval = wf - tar
    p = p+1
    panels[p].plot(diffval, 'k-', label='Output - Target', linewidth=lw)
    panels[p].set_ylabel('Output - Target')

    if (None == outfile):
        plt.show()
    else:
        # why save to pdf and not png? because most pdf viewers auto-reload on file change :-)
        #   (if you can find a good cross-platform image viewer that reloads, change the next line)
        plt.savefig(outfile+'.pdf')
        plt.close(fig)

    # save audio files
    sr = 44100
    st_audio.write_audio_file(outfile+'_input.wav', inp, sr)
    st_audio.write_audio_file(outfile+'_output.wav', wf, sr)
    st_audio.write_audio_file(outfile+'_target.wav', tar, sr)
    return


# when it's late & I'm watching the loss go down, things like this happen -SHH
def print_choochoo():
    print(" ~.~.~.~.      ")
    print(" ____    `.    ")
    print(" ]DD|_n_n_][   ")
    print(" |__|_______)  ")
    print(" 'oo OOOO oo\_ ")
    print("~+~+~+~+~+~+~+~")
    print("SignalTrain "+__version__)
    print("")

# EOF

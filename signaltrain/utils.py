
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


def progbar(epoch, max_epochs, batch_index, nbatches, loss, reg_term=None,
    width=30, vloss=None):
    """
    Simple progress bar for batch training.

    You can get a better ProgBar if you install something. ;-)
    Ripped off the style of Keras
    """
    base_str = '\rEpoch '+str(epoch)+' /'+str(max_epochs)+': '
    percent = (batch_index+1)/nbatches
    barlength = int(width * percent)
    bar  = '='*barlength + '>'
    leftover = width - barlength
    space =  ' '*leftover
    loss_num = loss#
    print(base_str + bar + space+' loss: {:10.5e}'.format(loss_num), end="")
    return


class LossLogger():
    """Class for keeping track of and printing loss info"""

    def __init__(self, name='run'):
        """
        name is the tag for the run. it will be appended with the current
        time & date.  a directory with this full name will be created, and
        logs will go there
        """
        self.history = {'Epoch':[], 'Train':[], 'Val':[]}#, 'Reg':[]}
        self.best_vloss = 999.9
        self.name = name + '_{date:%H:%M:%S_%b_%d_%Y}'.format( date=datetime.datetime.now() )
        self.lossfile = "losses.csv"
        self.setup_files()

    def setup_files(self):
        """opens loss-tracking directory for writing"""
        print('LossLogger: logging to directory "'+self.name+'"')
        os.mkdir(self.name)   # TODO: noclobber. Starting multiple simultaneous runs in the same directory
                              # with the same tag (unlikely) will fail here
        with open(self.name+'/'+self.lossfile, "w") as myfile:
            myfile.write('#Epoch, TrainLoss, ValTotalLoss, ValMSELoss, ValL1Loss\n')

    def update_file(self):
        """append loss info to (CSV) file"""
        with open(self.name+'/'+self.lossfile, "a+") as myfile:
            outstr = str(self.history['Epoch'][-1])+', {:10.5e}'.format(self.history['Train'][-1])+ \
                ', {:10.5e}'.format(self.history['Val'][-1])
            myfile.write(outstr+'\n')

    def update(self, epoch, loss, vloss):
        """the main interface; stores losses in history dict, updates file"""
        tloss, vloss = loss, vloss # loss.item(), vloss.item()
        self.history['Epoch'].append(epoch)
        self.history['Train'].append(tloss)
        self.history['Val'].append(vloss)
        print('  val_loss: {:10.5e}'.format(self.history['Val'][-1]))
        self.update_file()

    def is_best(self):
        """only updates best_vloss when called; this is intentional"""
        vloss_num = self.history['Val'][-1]  # most recent val loss
        if (vloss_num < self.best_vloss):
            self.best_vloss = vloss_num
            return True
        return False



def save_checkpoint(state, filename='checkpoint.pth.tar', best_only=False, is_best=False):
    """
    Saves the 'state' of model and optimization to a file

     state can include multiple elements in dict format, e.g.
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),'optimizer': optimizer.state_dict(),}
    """
    if is_best:
        print("Best val_loss so far (for checkpoint).    ", end="")

    if (not best_only) or is_best:
        print("Saving checkpoint in",filename)
        torch.save(state, filename)
    return


def load_checkpoint(model, optimizer, losslogger, filename='checkpoint.pth.tar'):
    """
    Loads from a checkpoint file and initializes model weights, optimizer params
         & lossloger parts

    Note: Input model & optimizer should be pre-defined.
    This routine only updates their states.
    """
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        #best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        savename = losslogger.name                # keep from overwriting current run name
        losslogger = checkpoint['losslogger']
        losslogger.name = savename
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger


def load_weights(model, filename='checkpoint.pth.tar'):
    """
    simpler than load_checkpoint(); this ignores everything but the model
    """
    if os.path.isfile(filename):
        print("=> Initializing model weights from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(filename))
    return model




def overlapping_windows(X, window_size, overlap):
    """
    Create an overlapped version of X
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    overlap : int
        Number of elements to overlap by on each side (zero pads on ends)
    Returns
    -------
    X_windows : shape=(n_windows, window_size)
        2D array of overlapped X

    So, for each window, the amount of 'unique' data will be (window_size - 2*overlap)

    Example:
    X = np.arange(15)
    X =  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]  (15 elements)
    overlapping_windows(X, 5, 2)
    X_padded = [ 0 0 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 0 0] (17 elements)
    X_windows = [[0 0 0 1 2]
                 [0 0 1 2 3]
                 [0 1 2 3 4]
                 [... 3 ...]
                 [    4    ]
                 [    5    ]
                 [    6    ]
                 [    7    ]
                 [    8    ]
                 [    9    ]
                 [    10    ]
                 [    11    ]
               [10 11 12 13 14]
               [11 12 13 14  0]
               [12 13 14  0 0]]
    overlapping_windows(X, 6, 2)
    X_windows =[[0 0 0 1 2 3]
                [0 1 2 3 4 5]
                [2 3 4 5 6 7]
                [4 5 6 7 8 9]
                [6 7 8 9 10 11]
                [8 9 10 11 12 13]
                [10 11 12 13 14 0]
                [11 12 14  0  0 0]]

    Initially inspired by Kyle Kastner's "overlap" https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
     but completely rewritten for my needs/specifications
    """
    assert window_size > 2*overlap, "Error, windowsize(="+str(window_size)+ \
        ") must be at least twice overlap(="+str(overlap)+')'

    # pad zeros on the front and rear equal to overlap
    X_padded = np.pad(X,(overlap,overlap),'constant',constant_values=0)

    # figure out info about new array
    unique_per_window = window_size - 2*overlap
    n_windows = int(np.ceil(X.shape[0] / unique_per_window))

    # now make sure X_padded is an integer multiple of window_size
    extra_pad = window_size - (X_padded.shape[0] % window_size)   # how much more we need to pad on end
    if (extra_pad != 0):
        X_padded = np.pad(X_padded,(0,extra_pad),'constant',constant_values=0)

    # now figure out the new shape for the new array
    strides = (X_padded.itemsize*(window_size - 2*overlap), X_padded.itemsize)
    X_windows = np.lib.stride_tricks.as_strided(X_padded, shape=(n_windows, window_size), strides=strides)
    return X_windows


def undo_overlapping_windows(X_windows, overlap):  # inverse of above
    """
    Inverse of overlapping_windows
    Parameters
    ----------
    X_windows: ndarray, shape=(n_windows, window_size)
        The array of windowed input signal
    overlap: int
         The amount of overlap that is to be undone
    Returns
    -------
    X_out: ndarray, shape=(signal_length)
    """
    num_windows = X_windows.shape[0]
    window_size = X_windows.shape[1]
    unique_per_window = window_size - 2*overlap
    X_out = X_windows[:,overlap: window_size-overlap]
    return X_out.ravel()


def sequential_windows(signal, window_size):
    """like overlapping windows, except each window advances by 1 sample"""
    shape = (signal.size - window_size + 1, window_size)
    strides = signal.strides * 2
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

def undo_sequential_windows(stack):
    """inverse of sequential_windows()"""
    window_size = stack.shape[-1]
    return np.concatenate((stack[0,0:window_size-1],stack[:,-1]))


# generic wrapper for different ways of converting 1-D signal to 2-D stack of windows
def chopnstack(signal, chunk_size=8192, overlap=0, dtype=np.float32):
    """
    Chop'n'Stack:  Cuts up a signal into chunks and stacks them vertically. Pad with zeros
    Parameters
    ----------
    sig:   signal, ndarray, shape=(batch_size, signal_length)
    """
    return sequential_windows(signal, chunk_size)
    if False:
        unique_per_chunk = chunk_size - 2*overlap
        num_chunks = int(np.ceil(sig.shape[1] / unique_per_chunk))


        batch_size = sig.shape[0]
        print("we think batch_size = ",batch_size)
        if (0 == overlap):  # if this can be fast & easy
            return np.reshape(sig, (batch_size, num_chunks, chunk_size))

        stack = np.zeros((batch_size, num_chunks, chunk_size),dtype=dtype)
        for bi in range(batch_size):
            stack[bi,:,:] = overlapping_windows(sig[bi,:], chunk_size, overlap)
        return stack


# Inverse of Chop'n'Stack: takes vertical stack and produces 1-D signal
def inv_chopnstack(stack, overlap=0, signal_length=None):
    """
    Inverse chopnstack
    Parameters
    ----------
    stack : ndarray, shape=(batch_size, num_chunks, chunk_size)
        Input signal to window and overlap
    overlap : int
        Number of elements that are overlapping on on each side (zero pads on ends)
    signal_length: int
        Specifying this will truncate the output signal(s) to remove any zero-padding
    Returns
    -------
    sig : signal, shape=(batch_size, signal_length)
    """
    return undo_sequential_windows(stack)
    if False:
        batch_size = stack.shape[0]
        if (0==overlap) and (1 == batch_size):
            return stack.ravel()

        if None==signal_length:
            signal_length = stack.shape[1]*stack.shape[2]

        sig = np.zeros((batch_size, signal_length),dtype=dtype)
        for bi in range(batch_index):
            sig[bi,:] = undo_overlapping_windows(stack[bi], overlap=overlap)[0:signal_length]
        return sig


def slice_prediction(stack,mode='last'):
    """
    Slices the output along some column.

    Called either for computing losses or for visualization
    """
    if ('all' == mode):
        return inv_chopnstack(stack)
    elif ('last' == mode):  # last column
        return stack[:,:,-1]
    elif ('center' == mode):
        ncols = stack.shape[-1]
        return stack[:,:,int(ncols/2)]
    else:
        raise(ValueError,"Invalid mode = '"+mode+"'")



def make_report(input_var, target_var, wave_form, loss_log, outfile=None,
    epoch=None, show_input=True, diff=False, mu_law=False):
    """
    Routine for printing progress of execution, e.g. graphs to files
    """
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


    panels[0].set_ylabel('Loss')
    panels[0].set_xlabel('Epoch')
    panels[0].legend(loc=1)
    xmin = 1
    if (ehist[-1] > 100):  # ignore the 1st 10 epochs (better when plotting with loglog scale)
        xmin = 100
    panels[0].set_xlim(left=xmin)
    panels[0].set_ylim(bottom=1e-4)#, top=loss_log.history['Val'][xmin] )


    if (input_var is None):  # exit, just show loss graph and get out
        plt.savefig(outfile+'.pdf')
        plt.close(fig)
        return

    # convert to 1D numpy arrays
    inp = slice_prediction(input_var.detach().cpu(), mode='center').numpy().T
    tar = slice_prediction(target_var.detach().cpu(), mode='center').numpy().T
    wf = slice_prediction(wave_form.detach().cpu(), mode='center').numpy().T
    #print("inp.shape = ",inp.shape)

    if mu_law:
        inp = st_audio.decode_mu_law(inp)
        tar = st_audio.decode_mu_law(tar)
        wf  = st_audio.decode_mu_law(wf)

    # middle panels: sample function(s)
    plot_len = int(len(inp))      # number of samples to show in the plot
    for signal_plot in range(num_signal_plots):
        p = signal_plot+1
        if (0==signal_plot):
            panels[p].plot(inp[0:plot_len], 'b-', label='Input', linewidth=lw)
        elif (signal_plot >=1):
            panels[p].plot(tar[0:plot_len], 'r-', label='Target', linewidth=lw)
        if (2==signal_plot):
            panels[p].plot(wf[0:plot_len], 'g-', label='Output', linewidth=lw)
        ampmax = 1.0 # 1.1*np.max(input_sig)
        panels[p].set_ylim((-ampmax,ampmax))   # for consistency
        panels[p].legend(loc=1)

    # panels[3]: plot difference between output and target
    diffval = wf - tar
    p = p+1
    panels[p].plot(diffval[0:plot_len], 'k-', label='Output - Target', linewidth=lw)
    panels[p].set_ylabel('Output - Target')

    if (None == outfile):
        plt.show()
    else:
        # Why save to PDF and not PNG? Because most PDF viewers auto-reload
        #   on file change :-)   If instead, if you can find a good
        #   cross-platform image viewer that reloads, change the next line
        plt.savefig(outfile+'.pdf')
        plt.close(fig)

    # save audio files
    sr = 44100
    # TODO: do we want to write the whole audio file?
    st_audio.write_audio_file(outfile+'_input.wav', inp, sr)
    st_audio.write_audio_file(outfile+'_output.wav', wf, sr)
    st_audio.write_audio_file(outfile+'_target.wav', tar, sr)
    return



def print_choochoo():
    """Just for fun. Makes a train picture."""
    print(" ~.~.~.~.      ")
    print(" ____    `.    ")
    print(" ]DD|_n_n_][   ")
    print(" |__|_______)  ")
    print(" 'oo OOOO oo\_ ")
    print("~+~+~+~+~+~+~+~")
    print("SignalTrain "+__version__)
    print("")

# EOF

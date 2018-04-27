#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Scott H. Hawley'
__copyright__ = 'Scott H. Hawley'

import numpy as np
import torch
import argparse
import signaltrain as st

from tensorboardX import SummaryWriter

def train_model(model, optimizer, criterion, X_train, Y_train,
    X_val=None, Y_val=None, losslogger=None, effect='ta',
    start_epoch=1, max_epochs=100,
    batch_size=100, sig_length=8192*100, fs=44100.,
    tol=1e-14, change_every=10, save_every=20, plot_every=20, retain_graph=False):

    if (losslogger is None):
        losslogger = st.utils.LossLogger()

    for epoch_iter in range(1, 1 + max_epochs - start_epoch): # start w/ 1 so can plot log/log scale
        epoch = start_epoch + epoch_iter

        # change the input dataset to encourage generality
        if (epoch > start_epoch) and (0 == epoch_iter % change_every):
            print("Preparing new data...")
            X_train, Y_train = st.audio.gen_audio(sig_length, chunk_size=int(X_train.size()[-1]), effect=effect, input_var=X_train, target_var=Y_train)

        print('Epoch ', epoch,' /', max_epochs, sep="", end=":")

        n_batches = 1#X_train.size()[0]
        for batch in range(n_batches):
            batch_index = batch * batch_size

            optimizer.zero_grad()                 # get ready to accumulate new gradients
            wave_form = model(X_train)            # run the neural network forward
            loss = criterion(wave_form, Y_train)  # score the output of forward inference

            if (X_val is None):   # we'll print both Trainin & validation losses later if we can
                losslogger.update(epoch, loss, None)

            if loss.data.cpu().numpy() < tol:
                break

            # TODO: move this back once we're doing 'real' batches
            if (n_batches > 1):
                loss.backward(retain_graph=retain_graph)
                optimizer.step()

        #----- after full dataset run through (all batches),do various reporting & diagnostic things

        if ((X_val is not None) and (Y_val is not None)):
            Ypred_val = model(X_val)
            vloss = criterion( Ypred_val, Y_val)
            losslogger.update(epoch, loss, vloss)

        if (epoch > start_epoch):
            device = str(torch.cuda.current_device())
            if (0 == epoch % save_every) or (epoch >= max_epochs-1):
                checkpoint_name = 'checkpoint'+device+'.pth.tar'
                st.utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                                           'optimizer': optimizer.state_dict(),
                                           'losslogger': losslogger,
                                           }, filename=checkpoint_name, best_only=True, is_best=losslogger.is_best() )

            if (0 == epoch % plot_every):
                outfile = 'progress'+device
                print("Writing progress report to ",outfile,'.*',sep="")
                st.utils.make_report(X_train, Y_train, wave_form, losslogger, outfile=outfile, epoch=epoch)
                st.models.model_viz(model,outfile)

        # TODO: move this back up if we get batches working properly
        if (1 == n_batches):
            loss.backward(retain_graph=False)#retain_graph)
            optimizer.step()

        if loss.data.cpu().numpy() < tol:  # need this 2nd break to get out of main Epoch loop
            break

    return model


# One additional model evaluation on Test or Val data
def eval_model(model, criterion, losslogger, sig_length, chunk_size, fs=44100., X=None, Y=None, effect='ta'): # X=input, Y=target
    print("\n\nEvaluating model: loss_num=",end="")
    if (None == X):
        X, Y = st.audio.gen_audio(sig_length, chunk_size=chunk_size, effect=effect)
    Ypred = model(X)    # run network forward
    loss = criterion(Ypred, Y)
    loss_num = loss.data.cpu().numpy()[0]
    print(loss_num)
    device = str(torch.cuda.current_device())
    outfile = 'final'+device
    print("Saving final Test evaluation report to ",outfile,".*",sep="")
    st.utils.make_report(X, Y, Ypred, losslogger, outfile=outfile)
    return


#-------------------------------------------------------------------------------
# MAIN BLOCK
#-------------------------------------------------------------------------------
def main():
    # set random seeds: useful for ensuring similar initializations when testing different (hyper)parameters
    torch.manual_seed(1)
    np.random.seed(1)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='trains SignalTrain effects mapper', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', default=10000, type=int, help="Number of iterations to train for")
    chunk_max = 15000     # roughly the maximum model size that will fit in memory on Titan X Pascal GPU
                          # if you immediately get OOM errors, decrease chunk_max
    parser.add_argument('--chunk', default=chunk_max, type=int, help="Length of each 'chunk' or window that input signal is chopped up into")
    parser.add_argument('--length', default=chunk_max*6000, type=int, help="Length of each audio signal (then cut up into chunks)")


    parser.add_argument('--fs', default=44100, type=int, help="Sample rate in Hertz")
    parser.add_argument('--device', default=0, type=int, help="CUDA device to use (e.g. 0 or 1)")
    parser.add_argument('--change', default=20, type=int, help="Changed data every this many epochs")
    parser.add_argument('--save', default=100, type=int, help="Save checkpoint (if best) every this many epochs")
    parser.add_argument('--plot', default=100, type=int, help="Plot report every this many epochs")
    parser.add_argument('--model', default='specsg', type=str,
                    help="Model type: 'specsg', 'spectral' or 'seq2seq'")

    # user can load a checkpoint file from a previous run: Either recall 'everything' (--resume) or just the weights (--init)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint.  Cannot be used with --init')
    group.add_argument('--init', default='', type=str, metavar='PATH',
                    help='initialize weights using checkpoint file. Like --resume only w/ default logger & optimizer')

    # audio effect to apply.    TODO: if effect name is not on the internal list, then check for a matching VST plugin via RenderMan
    parser.add_argument('--effect', default='ta', type=str,
                    help="Audio effect to try. Names are in signaltrain/audio.py. (default='ta' for time-alignment)")
    # TODO: decide how to specify paramaters to audio effects: individual values, or allow 'sweeps'?  Probably parse a string.

    # TODO: add command-line options for input & target audio:  Normally we synthesize everything, but we could:
    #    - Read lots of input-target pairs of audio waveforms  (in which case ---effect, above, should be 'turned off')
    #    - Read lots input audio, and then apply the desired effect
    #    - Assemble input (and target?) audio by pasting together short clips from a library of samples/clips (e.g. of drum sounds)



    args = parser.parse_args()
    print("args = ",args)
    sig_length = args.length
    fs = args.fs

    #--------------------
    # Check CUDA status
    #--------------------
    if torch.has_cudnn:
        torch.cuda.set_device(args.device)
        name = torch.cuda.get_device_name(args.device)
        print('Running on CUDA: device_count =',torch.cuda.device_count(),'. Using device ',args.device,':',name)
    else:
        print('No CUDA')

    #------------------------------------
    # Set up model and training criteria
    #------------------------------------
    retain_graph = False  # retain_graph only needed for RNN models. It's a memory hog
    if ('seq2seq' == args.model):
        model = st.models.Seq2Seq()
        retain_graph=True
    elif ('specsg' == args.model):
        model = st.models.SpecShrinkGrow(chunk_size=args.chunk)
    else:
        model = st.models.SpecEncDec(ft_size=args.chunk)
    model = torch.nn.DataParallel(model)       # run on multiple GPUs if possible

    losslogger = st.utils.LossLogger()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([ {'params': model.parameters()}], lr = 2e-4, eps=5e-6)#,  amsgrad=True)

    #-------------------------------------------------
    # Checkpoint recovery (if possible and requested) OR initialize just the weights
    #-------------------------------------------------
    start_epoch = 0
    if ('' != args.resume):
        model, optimizer, start_epoch, losslogger = st.utils.load_checkpoint(model, optimizer, losslogger, filename=args.resume)
    elif ('' != args.init):
        model = st.utils.load_weights(model, filename=args.init)   # why do this? because I want to try starting from 'autoencoder' weights

    #------------------------------------------------------------------
    # Once checkpoints are loaded (or not), CUDA-ify model if possible
    #------------------------------------------------------------------
    if torch.has_cudnn:
        model.cuda()

    #------------------------------------------
    # Set up Training and Validation datasets
    #------------------------------------------
    print("Peparing Training data...")
    X_train, Y_train = st.audio.gen_audio(sig_length, chunk_size=args.chunk, effect=args.effect)
    print("Peparing Validation data...")
    X_val, Y_val = st.audio.gen_audio(int(sig_length/5), chunk_size=args.chunk, effect=args.effect)

    #---------------
    # Training Loop
    #---------------
    model = train_model(model, optimizer, criterion, X_train, Y_train, X_val=X_val, Y_val=Y_val,
        start_epoch=start_epoch, max_epochs=args.epochs, losslogger=losslogger, effect=args.effect,
        fs=fs, sig_length=sig_length,
        save_every=args.save, change_every=args.change, plot_every=args.plot,
        retain_graph=retain_graph)

    #--------------------------------
    # Evaluate model on Test dataset
    #--------------------------------
    eval_model(model, criterion, losslogger,  sig_length, chunk_size=args.chunk, fs=fs, effect=args.effect)
    return

if __name__ == '__main__':

    main()

# EOF

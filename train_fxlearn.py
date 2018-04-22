#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Scott H. Hawley'
__copyright__ = 'Scott H. Hawley'

import numpy as np
import torch
import argparse
import fxlearn as fxl


def train_model(model, optimizer, criterion, X_train, Y_train,
    X_val=None, Y_val=None, losslogger=None,
    start_epoch=0, max_epochs=100,
    batch_size=100, sig_length=8192*100, fs=44100.,
    tol=1e-14, change_every=10, save_every=100, plot_every=10):

    if (losslogger is None):
        losslogger = fxl.utils.LossLogger()

    for epoch_iter in range(max_epochs - start_epoch):
        epoch = start_epoch + epoch_iter

        # change the input dataset to encourage generality
        if (epoch > start_epoch) and (0 == epoch_iter % change_every):
            print("Preparing new data...")
            X_train, Y_train = fxl.audio.gen_audio(sig_length, chunk_size=int(X_train.size()[-1]))

        print('Epoch ', epoch,' /', max_epochs, sep="", end=":")

        n_batches = 1#X_train.size()[0]
        for batch in range(n_batches):
            batch_index = batch * batch_size

            optimizer.zero_grad()
            wave_form = model(X_train)   # run the neural network forward
            loss = criterion(wave_form, Y_train)

            if (X_val is None):   # we'll print both losses later if we can
                losslogger.update(epoch, loss, None)

            if loss.data.cpu().numpy() < tol:
                break

            # TODO: move this back once we're doing 'real' batches
            if (n_batches > 1):
                loss.backward(retain_graph=True)      # retain_graph=True needed for RNNs, it seems
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
                print("Saving checkpoint in",checkpoint_name)
                fxl.utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                                           'optimizer': optimizer.state_dict(),
                                           'losslogger': losslogger,
                                           }, filename=checkpoint_name)
            if (0 == epoch % plot_every):
                outfile = 'progress'+device
                print("Saving progress report to ",outfile,'.*',sep="")
                fxl.utils.make_report(X_train, Y_train, wave_form, losslogger, outfile=outfile, epoch=epoch)
                fxl.models.model_viz(model,outfile)

        # TODO: move this back up if
        if (1 == n_batches):
            loss.backward(retain_graph=True)      # retain_graph=True needed for RNNs, it seems
            optimizer.step()

        if loss.data.cpu().numpy() < tol:  # need this 2nd break to get out of main Epoch loop
            break

    return model


# One additional model evaluation on Test or Val data
def eval_model(model, sig_length, fs=44100., X=None, Y=None): # X=input, Y=target
    print("\n\nEvaluating model: loss_num=",end="")
    if (None == X):
        X, Y = fxl.audio.gen_audio(sig_length)
    Ypred = model(X)    # run network forward
    loss = criterion(wave_form, Y)
    loss_num = loss.data.cpu().numpy()[0]
    print(loss_num)
    device = str(torch.cuda.current_device())
    outfile = 'final'+device
    print("Saving final Test evaluation report to ",outfile,".*",sep="")
    fxl.utils.make_report(X, Y, Ypred, loss_num, outfile=outfile)
    return


#-------------------------------------------------------------------------------
# MAIN BLOCK
#-------------------------------------------------------------------------------
def main():
    # useful for ensuring similar initializations when testing different models
    torch.manual_seed(1)
    np.random.seed(1)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='fxlearn')
    parser.add_argument('--epochs', default=10000, type=int, help="Number of iterations to train for")
    parser.add_argument('--length', default=8192*4000, type=int, help="Length of audio signals")
    parser.add_argument('--chunk', default=4096, type=int, help="Length of each 'chunk' or window that input signal is chopped up into")
    parser.add_argument('--plot', default=100, type=int, help="Plot report every this many epochs")

    parser.add_argument('--fs', default=44100, type=int, help="Sample rate in Hertz")
    parser.add_argument('--device', default=0, type=int, help="CUDA device to use (e.g. 0 or 1)")
    parser.add_argument('--change', default=50, type=int, help="Changed data every this many epochs")

    parser.add_argument('--model', default='spectral', type=str,
                    help="Model type: 'spectral' (default) or 'seq2seq'")
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
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
    if ('seq2seq' == args.model):
        model = fxl.models.Seq2Seq()
    else:
        model = fxl.models.SpecEncDec(ft_size=args.chunk)
    model = torch.nn.DataParallel(model)       # run on multiple GPUs if possible

    losslogger = fxl.utils.LossLogger()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([ {'params': model.parameters()}], lr = 0.001, eps=1e-7)#,  amsgrad=True)

    #-------------------------------------------------
    # Checkpoint recovery (if possible and requested)
    #-------------------------------------------------
    model, optimizer, start_epoch, losslogger = fxl.utils.load_checkpoint(model, optimizer, losslogger, filename=args.resume)

    #------------------------------------------------------------------
    # Once checkpoints are loaded (or not), CUDA-ify model if possible
    #------------------------------------------------------------------
    if torch.has_cudnn:
        model.cuda()

    #------------------------------------------
    # Set up Training and Validation datasets
    #------------------------------------------
    print("Peparing Training data...")
    X_train, Y_train = fxl.audio.gen_audio(sig_length, chunk_size=args.chunk)
    print("Peparing Validation data...")
    X_val, Y_val = fxl.audio.gen_audio(int(sig_length/8), chunk_size=args.chunk)

    #---------------
    # Training Loop
    #---------------
    model = train_model(model, optimizer, criterion, X_train, Y_train, X_val=X_val, Y_val=Y_val,
        start_epoch=start_epoch, max_epochs=args.epochs, losslogger=losslogger,
        fs=fs, sig_length=sig_length, change_every=args.change, plot_every=args.plot)

    #--------------------------------
    # Evaluate model on Test dataset
    #--------------------------------
    eval_model(model, sig_length, fs=fs)
    return

if __name__ == '__main__':

    main()

# EOF

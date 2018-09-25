#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Scott H. Hawley'
__copyright__ = 'Scott H. Hawley'
# this is the clean one

import numpy as np
import torch
import argparse
import signaltrain as st


def calc_loss(Y_pred, Y_true, criterion):
    """
    Function that calculates the loss function.
    Inputs:
      Y_pred: the output from the network
      Y_true: the target values
      criterion: (pointer to / name of) a PyTorch loss criterion, e.g. torch.nn.MSELoss()
    """
    # we may wish to slice the output, i.e. to ignore all but one time-slice
    mode = 'center'   # Note: 'center' can allow for non-causal inference
    ypred = st.utils.slice_prediction(Y_pred, mode=mode)
    ytrue = st.utils.slice_prediction(Y_true, mode=mode)

    return criterion(ypred, ytrue)


def train_model(model, optimizer, criterion, lambdas, train_gen, val_gen, cmd_args, device,
    losslogger=None, start_epoch=1, fs=44100,
    retain_graph=False, mu_law=False, tol=1e-14, dataset=None):
    """
    Main training routine
    """
    # Regularization term
    lambda_reg = 0.5

    # arguments from the command line
    (batch_size, max_epochs, effect, sig_length) = (cmd_args.batch,cmd_args.epochs,cmd_args.effect,cmd_args.length)
    (save_every, change_every, plot_every, model_name, chunk_size) = (cmd_args.save,cmd_args.change,cmd_args.plot,cmd_args.model,cmd_args.chunk)

    # setup logging routines
    if (losslogger is None):
        losslogger = st.utils.LossLogger()
    checkpoint_name = losslogger.name + '/checkpoint.pth.tar'
    progress_outfile = losslogger.name+'/progress'

    # Training loop of epochs
    for epoch_iter in range(1, 1 + max_epochs - start_epoch): # start w/ 1 so can plot log/log scale
        epoch = start_epoch + epoch_iter

        # generate new data every now & then
        if  (0 == epoch_iter % change_every) or (1==epoch_iter):
            print("   Changing to different input data")
            X_train, Y_train = next(train_gen)
            train_gen.send(True)    # tell generator to read from a new file next time
            X_val, Y_val = next(val_gen)
            Ypred_val = Y_val.clone()

        print('Epoch ', epoch,' /', max_epochs, sep="", end=":")

        # technically, batches are usually the '0' index; for now we're treating
        #      time steps as if they were batches, using index 1
        n_batches = int(X_train.size()[1] / batch_size)
        epoch_loss = 0
        for batch_num in range(n_batches):  # inside this loop is "per batch"

            optimizer.zero_grad()                 # get ready to accumulate new gradients

            # Send batches of data from CPU to GPU
            bgn, end = batch_num*batch_size, min( (batch_num+1)*batch_size, X_train.size()[1])
            X_batch = X_train[:,bgn:end,:].to(device)     # input signal,
            Y_batch = Y_train[:,bgn:end,:].to(device)     # target output

            # Predict output, calc loss, show progress bar
            Y_pred, layers, reg_term = model(X_batch)               # predict the output
            loss = calc_loss(Y_pred, Y_batch, criterion)
            epoch_loss += loss.item() + lambda_reg * reg_term.item()
            X_batch, Y_batch, Y_pred = X_batch.cpu(), Y_batch.cpu(), Y_pred.cpu()  # Move back to CPU (free up VRAM)
            st.utils.progbar(epoch, max_epochs, batch_num, n_batches, epoch_loss/(batch_num+1))  # show a progress bar through the epoch

            # Perform optimization (gradient descent) step
            loss.backward(retain_graph=retain_graph)    # usually retain_graph=False unless we use an RNN
            optimizer.step()

        epoch_loss /= n_batches    # not necessary but a nice feature: normalize loss by batch size

        #----- Now that a full dataset has been run through (all batches),
        #      Do various reporting & diagnostic things...

        # if we've been given Validation dataset
        if ((X_val is not None) and (Y_val is not None)):
            with torch.no_grad():           # not tracking gradients saves VRAM, bigtime
                n_batches = int(X_val.size()[1] / batch_size)
                epoch_vloss = 0
                for batch_num in range(n_batches):
                    #X_val, Y_val = X_val.to(device), Y_val.to(device)
                    bgn, end = batch_num*batch_size, min( (batch_num+1)*batch_size, X_val.size()[1])
                    X_batch = X_val[:,bgn:end,:].to(device)     # input signal,
                    Y_batch = Y_val[:,bgn:end,:].to(device)     # target output
                    Ypred_val_batch, layers_pred, reg_term = model(X_batch)
                    vloss = calc_loss(Ypred_val_batch, Y_batch, criterion)
                    epoch_vloss += vloss.item() + lambda_reg * reg_term.item()
                    Ypred_val[:,bgn:end,:] = Ypred_val_batch.detach().cpu()  # save the prediction for plotting later
                    #X_val, Y_val, Ypred_val = X_val.cpu(), Y_val.cpu(), Ypred_val.cpu()  # free up VRAM on the GPU
                epoch_vloss /= n_batches
            losslogger.update(epoch, epoch_loss, epoch_vloss)

        if (epoch > start_epoch):
            if (0 == epoch % save_every) or (epoch >= max_epochs-1):
                st.utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'losslogger': losslogger,},
                    filename=checkpoint_name, best_only=True, is_best=losslogger.is_best() )

            if (0 == epoch % plot_every):
                print("Writing progress report to ",progress_outfile,'.*',sep="")
                st.utils.make_report(X_val, Y_val, Ypred_val, losslogger, outfile=progress_outfile, epoch=epoch, mu_law=mu_law)
                st.models.model_viz(model,progress_outfile)

    return model



def main():
    """-------------------------------------------------------------------------------
                                   MAIN BLOCK
    -------------------------------------------------------------------------------"""

    # set random seeds: useful for ensuring similar initializations when testing different (hyper)parameters
    torch.manual_seed(0)
    np.random.seed(0)


    #-------------------------------
    # Parse command line arguments
    #-------------------------------
    # define a few defaults
    chunk_max = 12000      # Size of each window passed to model. The longer, the more interesting things you can do in the time domain
                          #    Note: chunk_max should be at least 3x as long as the compressor attack time (in samples)
                          #    chunk_max = 12000 is roughly the maximum model size that will fit in memory on GTX 1080 GPU
                          #    If you immediately get Out of Memory errors, decrease chunk_max
    batch_size = 100      # one big batch (e.g. 8000) runs faster per epoch and produces smoother loss curves,
                          #      but smaller batches converge faster & generalize better
    length = chunk_max * 2  # total length of each input signal

    parser = argparse.ArgumentParser(description='trains SignalTrain effects mapper', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch', default=batch_size, type=int, help="Batch size")  # TODO: get batch_size>1 working
    parser.add_argument('--change', default=10, type=int, help="Changed data every this many epochs")
    parser.add_argument('--chunk', default=chunk_max, type=int, help="Length of each 'chunk' or window that input signal is chopped up into")
    parser.add_argument('--effect', default='comp', type=str,
                    help="Audio effect to try. Names are in signaltrain/audio.py. (default='comp' for compressor)")
    parser.add_argument('--epochs', default=10000, type=int, help="Number of iterations to train for")
    parser.add_argument('--fs', default=44100, type=int, help="Sample rate in Hertz")
    parser.add_argument('--lambdas', default='0.0,0.0', type=str,
                    help="Comma-separated list of regularization penalty parameters, (L1, L2)")
    parser.add_argument('--length', default=length, type=int, help="Length of each audio signal (then cut up into chunks)")
    parser.add_argument('--lr', default=3e-4, type=float, help="Initial learning rate")
    parser.add_argument('--model', choices=['specsg'],
                    default='specsg', type=str, help="Choice of model to use")
    parser.add_argument('--name', default='run', type=str, help="Name/tag for this run. Logging goes into dir with this name. ")
    #TODO: decision: should it overwrite existing directory names, or create a new dir with, e.g. "_1" to avoid overwites?
    #      Answer: neither. We append the time & date of the run start to the name of the run, so they're all unique
    parser.add_argument("--parallel", help="Run in data-parallel mode",action="store_true")  # default=False
    parser.add_argument('--plot', default=10, type=int, help="Plot report every this many epochs")
    parser.add_argument('--save', default=20, type=int, help="Save checkpoint (if best) every this many epochs")

    # user can load a checkpoint file from a previous run: Either recall 'everything' (--resume) or just the weights (--init)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint.  Cannot be used with --init')
    group.add_argument('--init', default='', type=str, metavar='PATH',
                    help='initialize weights w/ checkpoint file. Like --resume only w/o logger & optimizer')

    # TODO: decide how to specify paramaters to audio effects: individual values, or allow 'sweeps'?  Probably parse a string.

    # parse / clean up command line arguments
    args = parser.parse_args()
    print("args = ",args)
    sig_length = args.length
    fs = args.fs
    lambdas = [float(x) for x in args.lambdas.split(',')] # turn comma-separated string into list of floats
    args.lr = args.lr / args.batch                        # found this to be necessary

    st.utils.print_choochoo()   # display program info

    #------------------------------
    # Check CUDA/CPU device status
    #------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # from pytorch 0.4 migration guide
    print("Running on device",device)


    #------------------------------------
    # Set up model and training criterion
    #------------------------------------
    retain_graph = False  # retain_graph only needed for RNN models. It's a memory hog
    mu_law = False # with mu-law companding, not only do we scale the floats into ints,
                  #  but the regression model should be turned into a multi-class classifier
                  # which means the target & output 'Y' values should get one-hot encoded

    if ('specsg' == args.model):
        model = st.models.SpecShrinkGrow_catri_skipadd(chunk_size=args.chunk)
    else:
        raise ValueError("specsg is the only current model")


    #----------------------------------------------------------------
    #  Set additional training specs based on which model was chosen
    #----------------------------------------------------------------

    losslogger = st.utils.LossLogger(name=args.name)
    criterion =  torch.nn.MSELoss()
    optimizer = torch.optim.Adam( model.parameters(), lr=args.lr, eps=5e-6)    #,  amsgrad=True)
    #-------------------------------------------------------------------
    # Checkpoint recovery (if requested) OR initialize just the weights
    #-------------------------------------------------------------------
    start_epoch = 0
    if ('' != args.resume):  # load weights and omptimizer settings, etc.
        model, optimizer, start_epoch, losslogger = st.utils.load_checkpoint(model, optimizer, losslogger, filename=args.resume)
    elif ('' != args.init):  # just load weights
        model = st.utils.load_weights(model, filename=args.init)   # why do this? because I want to try starting from 'autoencoder' weights

    #------------------------------------------------------------------
    # Once checkpoints are loaded (or not), CUDA-ify model if possible
    #------------------------------------------------------------------
    model = model.to(device)

    # For pytorch 0.4, manually move optimizer parts to GPU as well.  From https://github.com/pytorch/pytorch/issues/2830
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    #------------------------------------------
    # Set up Training and Validation data generators
    #------------------------------------------
    train_gen = st.audio.gen_audio(seq_size=int(args.length), chunk_size=args.chunk,path='Train',effect=args.effect)
    val_gen = st.audio.gen_audio(seq_size=int(args.length), chunk_size=args.chunk,path='Val', effect=args.effect, random_every=False)

    #--------------------
    # Call training Loop
    #--------------------
    model = train_model(model, optimizer, criterion, lambdas, train_gen, val_gen, args, device,
        start_epoch=start_epoch, losslogger=losslogger, fs=fs,
        retain_graph=retain_graph, mu_law=mu_law)

    return


if __name__ == '__main__':
    main()

# EOF

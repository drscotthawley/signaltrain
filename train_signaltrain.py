#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Scott H. Hawley'
__copyright__ = 'Scott H. Hawley'

import numpy as np
import torch
import argparse
import signaltrain as st

#from tensorboardX import SummaryWriter


def calc_loss(Y_pred, Y_true, criteria, lambdas):
    #  Y_pred: the output from the network
    #  Y_true: the target values
    #  criteria: an iterable of pytorch loss criteria, e.g. [torch.nn.MSELoss(), torch.nn.L1Loss()]
    #  lambdas:  a list of regularization parameters
    assert len(criteria) == len(lambdas), "Must have the same number of criteria as lambdas"

    for i in range(len(criteria)):
        if (0==i):
            loss = lambdas[i] * criteria[i](Y_pred, Y_true)
        else:
            loss += lambdas[i] * criteria[i](Y_pred, Y_true)
    return loss


def train_model(model, optimizer, criteria, lambdas, X_train, Y_train, cmd_args,
    X_val=None, Y_val=None, losslogger=None, start_epoch=1, fs=44100,
    retain_graph=False, mu_law=False, tol=1e-14, dataset=None):

    # arguments from the command line
    batch_size = cmd_args.batch
    max_epochs = cmd_args.epochs
    effect = cmd_args.effect
    sig_length= cmd_args.length
    save_every = cmd_args.save
    change_every = cmd_args.change
    plot_every = cmd_args.plot
    model_name = cmd_args.model
    chunk_size = cmd_args.chunk

    print("X_train.size() =",X_train.size(),", batch_size =",batch_size)
    if (0 != X_train.size()[0] % batch_size):
        raise ValueError("X_train.size()[0] = ", X_train.size()[0],
            ", must be an integer multiple of batch_size =",batch_size)
    nbatches =  int(X_train.size()[0] / batch_size)
    print(nbatches,"batches per epoch")

    if (dataset is not None):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    if (losslogger is None):
        losslogger = st.utils.LossLogger()

    for epoch_iter in range(1, 1 + max_epochs - start_epoch): # start w/ 1 so can plot log/log scale
        epoch = start_epoch + epoch_iter

        # re-generate the input dataset occasionally, to encourage generality
        if (epoch > start_epoch) and (0 == epoch_iter % change_every):
            print("Preparing new data...")
            X_train, Y_train = st.audio.gen_audio(sig_length, chunk_size=int(X_train.size()[-1]),
                effect=effect, input_var=X_train, target_var=Y_train)

        # note: we print the Epoch after the new data just to keep the console display 'consistent'
        print('Epoch ', epoch,' /', max_epochs, sep="", end=":")

        # Batch training with dataloader.... not quite working yet
        #for ibatch, data in enumerate(dataloader):   # batches
        #    X_train_batch, X_train_batch =  data['input'], data['target']

        for bi in range(nbatches):
            bstart = bi * batch_size
            X_train_batch = X_train[bstart : bstart + batch_size, :]
            Y_train_batch = Y_train[bstart : bstart + batch_size, :]

            optimizer.zero_grad()                 # get ready to accumulate new gradients

            # forward + backward + optimize
            wave_form = model(X_train_batch)
            loss = calc_loss(wave_form, Y_train_batch, criteria, lambdas)
            loss.backward(retain_graph=retain_graph)    # usually retain_graph=False unless we use an RNN
            optimizer.step()

            st.utils.progbar(epoch, max_epochs, bi, nbatches, loss)  # show a progress bar through the epoch

        if loss.data.cpu().numpy() < tol:
            break

        #----- after full dataset run through (all batches),do various reporting & diagnostic things

        if ((X_val is not None) and (Y_val is not None)):
            Ypred_val = model(X_val)
            vloss = calc_loss(Ypred_val, Y_val, criteria, lambdas)
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
                st.utils.make_report(X_val, Y_val, Ypred_val, losslogger, outfile=outfile, epoch=epoch, mu_law=mu_law)
                st.models.model_viz(model,outfile)

    return model


# One additional model evaluation on Test or Val data
def eval_model(model, criterion, losslogger, sig_length, chunk_size, fs=44100.,
    X=None, Y=None, effect='ta', mu_law=False): # X=input, Y=target
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
    st.utils.make_report(X, Y, Ypred, losslogger, outfile=outfile, mu_law=mu_law)
    return


#-------------------------------------------------------------------------------
#                                MAIN BLOCK
#-------------------------------------------------------------------------------
def main():

    # set random seeds: useful for ensuring similar initializations when testing different (hyper)parameters
    torch.manual_seed(1)
    np.random.seed(1)


    #-------------------------------
    # Parse command line arguments
    #-------------------------------
    chunk_max = 15000     # roughly the maximum model size that will fit in memory on Titan X Pascal GPU
                          # if you immediately get OOM errors, decrease chunk_max
    batch_size = 500      # one big batch (e.g. 8000) runs faster per epoch, but smaller batches converge faster
    length = chunk_max * 10000

    parser = argparse.ArgumentParser(description='trains SignalTrain effects mapper', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch', default=batch_size, type=int, help="Batch size")  # TODO: get batch_size>1 working
    parser.add_argument('--change', default=20, type=int, help="Changed data every this many epochs")
    parser.add_argument('--chunk', default=chunk_max, type=int, help="Length of each 'chunk' or window that input signal is chopped up into")
    parser.add_argument('--device', default=0, type=int, help="CUDA device to use (e.g. 0 or 1)")
    parser.add_argument('--effect', default='ta', type=str,
                    help="Audio effect to try. Names are in signaltrain/audio.py. (default='ta' for time-alignment)")
    parser.add_argument('--epochs', default=10000, type=int, help="Number of iterations to train for")
    parser.add_argument('--fs', default=44100, type=int, help="Sample rate in Hertz")
    parser.add_argument('--length', default=length, type=int, help="Length of each audio signal (then cut up into chunks)")
    #parser.add_argument('--lr', default=2e-4, type=float, help="Initial learning rate")
    parser.add_argument('--model', choices=['specsg','spectral','seq2seq','wavenet'], default='specsg', type=str,
                    help="Name of model lto use")
    parser.add_argument('--plot', default=20, type=int, help="Plot report every this many epochs")
    parser.add_argument('--save', default=20, type=int, help="Save checkpoint (if best) every this many epochs")

    # user can load a checkpoint file from a previous run: Either recall 'everything' (--resume) or just the weights (--init)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint.  Cannot be used with --init')
    group.add_argument('--init', default='', type=str, metavar='PATH',
                    help='initialize weights w/ checkpoint file. Like --resume only w/o logger & optimizer')

    # TODO: decide how to specify paramaters to audio effects: individual values, or allow 'sweeps'?  Probably parse a string.

    # TODO: add command-line options for input & target audio:  Normally we synthesize everything, but we could:
    #    - Read lots of input-target pairs of audio waveforms  (in which case ---effect, above, should be 'turned off')
    #    - Read lots input audio, and then apply the desired effect
    #    - Assemble input (and target?) audio by pasting together short clips from a library of samples/clips (e.g. of drum sounds)

    args = parser.parse_args()
    print("args = ",args)
    sig_length = args.length
    fs = args.fs

    st.utils.print_choochoo()   # display program info

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
    mu_law = False # with mu-law companding, not only do we scale the floats into ints,
                  #  but the regression model should be turned into a multi-class classifier
                  # which means the target & output 'Y' values should get one-hot encoded

    if ('seq2seq' == args.model):
        model = st.models.Seq2Seq()
        retain_graph=True
    elif ('specsg' == args.model):
        model = st.models.SpecShrinkGrow(chunk_size=args.chunk)
    elif ('wavenet' == args.model):
        model = st.models.WaveNet()
        mu_law = True
    else:
        model = st.models.SpecEncDec(ft_size=args.chunk)

    model = torch.nn.DataParallel(model)       # run on multiple GPUs if possible

    #----------------------------------------------------------------
    #  Set additional training specs based on which model was chosen
    #----------------------------------------------------------------
    losslogger = st.utils.LossLogger()

    if (args.model != 'wavenet'):
        criteria = [torch.nn.MSELoss(), torch.nn.L1Loss()]
        lambdas = [0.8, 0.2]
        optimizer = torch.optim.Adam( model.parameters(), lr=2e-4, eps=5e-6)    #,  amsgrad=True)
    else:
        raise ValueError("Sorry, model 'wavenet' isn't ready yet.")
        criteria = [torch.nn.CrossEntropyLoss()]
        lambdas = [1.0]
        optimizer = torch.optim.Adam( model.parameters(), lr=0.01)


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
    if torch.has_cudnn:
        model.cuda()

    #------------------------------------------
    # Set up Training and Validation datasets
    #------------------------------------------
    print("Peparing Training data...")
    dataset = None#  st.audio.AudioDataset(args.length, chunk_size=args.chunk, effect=args.effect)
    X_train, Y_train = st.audio.gen_audio(sig_length, chunk_size=args.chunk, effect=args.effect, mu_law=mu_law)
    print("Peparing Validation data...")
    X_val, Y_val = st.audio.gen_audio(int(sig_length/5), chunk_size=args.chunk, effect=args.effect, mu_law=mu_law, x_grad=False)

    #--------------------
    # Call training Loop
    #--------------------
    model = train_model(model, optimizer, criteria, lambdas, X_train, Y_train, args,
        X_val=X_val, Y_val=Y_val, start_epoch=start_epoch, losslogger=losslogger, fs=fs,
        retain_graph=retain_graph, mu_law=mu_law, dataset=dataset)

    #--------------------------------
    # Evaluate model on Test dataset
    #--------------------------------
    eval_model(model, criterion, losslogger, sig_length, chunk_size=args.chunk, fs=fs, effect=args.effect, mu_law=mu_law)
    return

if __name__ == '__main__':

    main()

# EOF

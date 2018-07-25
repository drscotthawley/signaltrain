#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Scott H. Hawley'
__copyright__ = 'Scott H. Hawley'

import numpy as np
import torch
import argparse
import signaltrain as st


def calc_loss(Y_pred, Y_true, criterion):
    #  Y_pred: the output from the network
    #  Y_true: the target values
    #  criterion: an pytorch loss criterion, e.g. torch.nn.MSELoss()

    # we may wish to slice the output, i.e. to ignore all but one time-slice
    ypred = st.utils.slice_prediction(Y_pred, mode='center')  # 'center' can allow for non-causal inference
    ytrue = st.utils.slice_prediction(Y_true, mode='center')

    return criterion(ypred, ytrue)


def train_model(model, optimizer, criterion, lambdas, train_gen, val_gen, cmd_args, device,
    losslogger=None, start_epoch=1, fs=44100,
    retain_graph=False, mu_law=False, tol=1e-14, dataset=None):

    # arguments from the command line
    (batch_size, max_epochs, effect, sig_length) = (cmd_args.batch,cmd_args.epochs,cmd_args.effect,cmd_args.length)
    (save_every, change_every, plot_every, model_name, chunk_size) = (cmd_args.save,cmd_args.change,cmd_args.plot,cmd_args.model,cmd_args.chunk)

    if (losslogger is None):
        losslogger = st.utils.LossLogger()
    checkpoint_name = losslogger.name + '/checkpoint.pth.tar'
    progress_outfile = losslogger.name+'/progress'

    for epoch_iter in range(1, 1 + max_epochs - start_epoch): # start w/ 1 so can plot log/log scale
        epoch = start_epoch + epoch_iter

        # generate new data every now & then
        if  (0 == epoch_iter % change_every) or (1==epoch_iter):
            X_train, Y_train = next(train_gen)
            train_gen.send(True)    # tell generator to read from a new file next time
            X_val, Y_val = next(val_gen)
            Ypred_val = Y_val.clone()

        print('Epoch ', epoch,' /', max_epochs, sep="", end=":")

        # technically, batches are usually the '0' index; for now we're treating
        # time steps as if they were batches, using index 1
        n_batches = int(X_train.size()[1] / batch_size)
        epoch_loss = 0
        for batch_num in range(n_batches):  # inside this loop is "per batch"

            optimizer.zero_grad()                 # get ready to accumulate new gradients

            # to avoid CUDA out of memory, only send batches to the GPU
            bgn, end = batch_num*batch_size, min( (batch_num+1)*batch_size, X_train.size()[1])
            X_batch = X_train[:,bgn:end,:].to(device)     # input signal,
            Y_batch = Y_train[:,bgn:end,:].to(device)     # target output
            Y_pred, layers = model(X_batch)               # predicted output

            # A solution needs to be found in order to obtain a variable from the model here.
            # Let's call it "z" for a moment.
            if len(layers) > 0:
                z = layers[0]
                reg_term = z.norm(1) * 1e-4

            loss = calc_loss(Y_pred, Y_batch, criterion)
            epoch_loss += loss.item()
            X_batch, Y_batch, Y_pred = X_batch.cpu(), Y_batch.cpu(), Y_pred.cpu()  # free up VRAM
            st.utils.progbar(epoch, max_epochs, batch_num, n_batches, epoch_loss/(batch_num+1))  # show a progress bar through the epoch

            #reg_term = loss * 1e-4 # Dumb regularization for the moment

            # Apply regularization to loss
            # loss += reg_term

            # Perform optimization / gradient descent
            loss.backward(retain_graph=retain_graph)    # usually retain_graph=False unless we use an RNN
            optimizer.step()

        #if loss.data.cpu().numpy() < tol:   # commented out: if it hits a patch of zeros, we don't exit
        #    break
        epoch_loss /= n_batches

        #----- after full dataset run through (all batches),do various reporting & diagnostic things


        if ((X_val is not None) and (Y_val is not None)):
            with torch.no_grad():           # not tracking gradients saves VRAM, bigtime
                n_batches = int(X_val.size()[1] / batch_size)
                epoch_vloss = 0
                for batch_num in range(n_batches):
                    #X_val, Y_val = X_val.to(device), Y_val.to(device)
                    bgn, end = batch_num*batch_size, min( (batch_num+1)*batch_size, X_val.size()[1])
                    X_batch = X_val[:,bgn:end,:].to(device)     # input signal,
                    Y_batch = Y_val[:,bgn:end,:].to(device)     # target output
                    Ypred_val_batch, layers_pred = model(X_batch)
                    vloss = calc_loss(Ypred_val_batch, Y_batch, criterion)
                    epoch_vloss += vloss.item()
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


# One additional model evaluation on Test or Val data
def eval_model(model, criterion, lambdas, losslogger, sig_length, chunk_size, device, effect,
    fs=44100, X=None, Y=None, mu_law=False):
    # X_test=input, Y_test=target (optional_cal be regenerated)
    print("\n\nEvaluating model: loss_num=",end="")
    if (None == X):
        X_test, Y_test = st.audio.gen_audio(int(sig_length/5), device, chunk_size=chunk_size,
            effect=effect, input_var=X_train, target_var=Y_train)
        # st.audio.gen_audio(sig_length, chunk_size=chunk_size, effect=effect)
    Y_pred, layers_pred = model(X)    # run network forward
    loss = calc_loss(Y_pred, Y_test, criterion, lambdas)
    loss_num = loss.data.cpu().numpy()
    print(loss_num)
    outfile = 'final'
    print("Saving final Test evaluation report to ",outfile,".*",sep="")
    st.utils.make_report(X, Y, Ypred, losslogger, outfile=outfile, mu_law=mu_law)
    return


#-------------------------------------------------------------------------------
#                                MAIN BLOCK
#-------------------------------------------------------------------------------
def main():

    # set random seeds: useful for ensuring similar initializations when testing different (hyper)parameters
    torch.manual_seed(0)
    np.random.seed(0)


    #-------------------------------
    # Parse command line arguments
    #-------------------------------
    chunk_max = 15000     # roughly the maximum model size that will fit in memory on Titan X Pascal GPU
                          # if you immediately get OOM errors, decrease chunk_max
    batch_size = 100      # one big batch (e.g. 8000) runs faster per epoch, but smaller batches converge faster
    fs = 44100
    length = fs * 5     # 5 seconds of audio

    parser = argparse.ArgumentParser(description='trains SignalTrain effects mapper', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch', default=batch_size, type=int, help="Batch size")  # TODO: get batch_size>1 working
    parser.add_argument('--change', default=20, type=int, help="Changed data every this many epochs")
    parser.add_argument('--chunk', default=chunk_max, type=int, help="Length of each 'chunk' or window that input signal is chopped up into")
    parser.add_argument('--effect', default='comp', type=str,
                    help="Audio effect to try. Names are in signaltrain/audio.py. (default='ta' for time-alignment)")
    parser.add_argument('--epochs', default=10000, type=int, help="Number of iterations to train for")
    parser.add_argument('--fs', default=fs, type=int, help="Sample rate in Hertz")
    parser.add_argument('--lambdas', default='0.0,0.0', type=str,
                    help="Comma-separated list of regularization penalty parameters, (L1, L2)")
    parser.add_argument('--length', default=length, type=int, help="Length of each audio signal (then cut up into chunks)")
    parser.add_argument('--lr', default=1e-6, type=float, help="Initial learning rate")
    parser.add_argument('--model', choices=['specsg','spectral','seq2seq','wavenet','specfb','manyto1'],
                    default='specsg', type=str, help="Choice of model lto use")
    parser.add_argument('--name', default='run', type=str, help="Name/tag for this run. Logging goes into dir with this name. ")
    #TODO: decision: should it overwrite existing directory names, or create a new dir with, e.g. "_1" to avoid overwites?
    #      Answer: neither. We append the time & date of the run start to the name of the run, so they're all unique
    parser.add_argument("--parallel", help="Run in data-parallel mode",action="store_true")  # default=False
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
    lambdas = [float(x) for x in args.lambdas.split(',')] # turn comma-sep string into list of floats

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

    if ('seq2seq' == args.model):
        model = st.models.Seq2Seq()
        retain_graph=True
    elif ('specsg' == args.model):
        model = st.models.SpecShrinkGrow_catri_skipadd(chunk_size=args.chunk)
    elif ('specfb' == args.model):
        model = st.models.SpecFrontBack(chunk_size=args.chunk)
    elif ('manyto1' == args.model):
        model = st.models.FNNManyToOne(chunk_size=args.chunk)  # terrible
    elif ('wavenet' == args.model):
        model = st.models.WaveNet()
        mu_law = True
    else:
        model = st.models.SpecEncDec(ft_size=args.chunk)

    # Default is serial, not data-parallel execution. But --parallel flag will enable DP
    #   In my experiments, DataParallel runs no faster than single-GPU for the same
    #   batch size.  It will allow you to run with larger batches, but...that's it.
    if args.parallel and (torch.cuda.device_count() > 1):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    #----------------------------------------------------------------
    #  Set additional training specs based on which model was chosen
    #----------------------------------------------------------------
    losslogger = st.utils.LossLogger(name=args.name)

    if (args.model != 'wavenet'):
        criterion =  torch.nn.MSELoss()
        optimizer = torch.optim.Adam( model.parameters(), lr=args.lr, eps=5e-6)    #,  amsgrad=True)
    else:
        raise ValueError("Sorry, model 'wavenet' isn't ready yet.")
        criterion = torch.nn.CrossEntropyLoss()
        lambdas = [1.0]  # replace any args.lambdas value because there's only one loss in wavenet
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

    #--------------------------------
    # Evaluate model on Test dataset
    #--------------------------------
    # Broken for now
    #eval_model(model, criteria, lambdas, losslogger, sig_length, chunk_size, device, args.effect,
    #    fs=fs, effect=args.effect, mu_law=mu_law, dataset=dataset)
    return

if __name__ == '__main__':
    main()

# EOF

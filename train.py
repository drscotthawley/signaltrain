#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Scott H. Hawley'
__copyright__ = 'Scott H. Hawley'

import numpy as np
import torch
import argparse
import signaltrain as st

#from tensorboardX import SummaryWriter


def l1_penalty(var):
    return torch.abs(var).mean()

def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).mean())

def calc_loss(Y_pred, Y_true, criteria, lambdas):
    #  Y_pred: the output from the network
    #  Y_true: the target values
    #  criteria: an iterable of pytorch loss criteria, e.g. [torch.nn.MSELoss(), torch.nn.L1Loss()]
    #  lambdas:  a list of regularization parameters
    numlambdas = len(lambdas)
    loss_list = [0]*numlambdas   # report each of the loss terms (with the lambda multiplication included)
    loss_list[0] = lambdas[0]*l1_penalty(Y_pred)
    loss_list[1] = lambdas[1]*l2_penalty(Y_pred)
    loss = criteria[0](Y_pred, Y_true)
    for i in range(numlambdas):
        loss += loss_list[i]
    return loss, loss_list


def train_model(model, optimizer, criteria, lambdas, X_train, Y_train, cmd_args, device,
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
            X_train, Y_train = st.audio.gen_audio(sig_length, batch_size, device, chunk_size=int(X_train.size()[-1]),
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
            loss, losses = calc_loss(wave_form, Y_train_batch, criteria, lambdas)
            loss.backward(retain_graph=retain_graph)    # usually retain_graph=False unless we use an RNN
            optimizer.step()

            st.utils.progbar(epoch, max_epochs, bi, nbatches, loss)  # show a progress bar through the epoch

        if loss.data.cpu().numpy() < tol:
            break

        #----- after full dataset run through (all batches),do various reporting & diagnostic things

        if ((X_val is not None) and (Y_val is not None)):
            Ypred_val = model(X_val)
            # On the next line, lambdas=[0,0]: don't include regularization terms on val set evaluation
            vloss, vlosses = calc_loss(Ypred_val, Y_val, criteria, [0,0])
            losslogger.update(epoch, loss, vloss, vlosses)

        if (epoch > start_epoch):
            if (0 == epoch % save_every) or (epoch >= max_epochs-1):
                checkpoint_name = losslogger.name + '/checkpoint.pth.tar'
                st.utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                                           'optimizer': optimizer.state_dict(),
                                           'losslogger': losslogger,
                                           }, filename=checkpoint_name,
                                           best_only=True, is_best=losslogger.is_best() )

            if (0 == epoch % plot_every):
                outfile = losslogger.name+'/progress'
                print("Writing progress report to ",outfile,'.*',sep="")
                st.utils.make_report(X_val, Y_val, Ypred_val, losslogger, outfile=outfile, epoch=epoch, mu_law=mu_law)
                st.models.model_viz(model,outfile)

    return model


# One additional model evaluation on Test or Val data
def eval_model(model, criteria, lambdas, losslogger, sig_length, chunk_size, device, effect,
    fs=44100, X=None, Y=None, mu_law=False):
    # X_test=input, Y_test=target (optional_cal be regenerated)
    print("\n\nEvaluating model: loss_num=",end="")
    if (None == X):
        X_test, Y_test = st.audio.gen_audio(int(sig_length/5), device, chunk_size=chunk_size,
            effect=effect, input_var=X_train, target_var=Y_train)
        # st.audio.gen_audio(sig_length, chunk_size=chunk_size, effect=effect)
    Y_pred = model(X)    # run network forward
    loss = calc_loss(Y_pred, Y_test, criteria, lambdas)
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
    torch.manual_seed(1)
    np.random.seed(1)


    #-------------------------------
    # Parse command line arguments
    #-------------------------------
    chunk_max = 8192     # roughly the maximum model size that will fit in memory on Titan X Pascal GPU
                          # if you immediately get OOM errors, decrease chunk_max
    batch_size = 200      # one big batch (e.g. 8000) runs faster per epoch, but smaller batches converge faster
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
    parser.add_argument('--lr', default=2e-4, type=float, help="Initial learning rate")
    parser.add_argument('--model', choices=['specsg','spectral','seq2seq','wavenet'], default='specsg', type=str,
                    help="Type of model lto use")
    parser.add_argument('--name', default='run', type=str, help="Name/tag for this run. Data/logging goes into dir with this name. ")
    #TODO: decision: should it overwrite existing directory names, or create a new dir with, e.g. "_1" to avoid overwites?
    #      Answer: neither. We append the time & date of the run start to the name of the run, so they're all unique
    parser.add_argument("--parallel", help="Run in data-parallel mode",action="store_true")
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
        model = st.models.SpecShrinkGrow_cat(chunk_size=args.chunk)
    elif ('wavenet' == args.model):
        model = st.models.WaveNet()
        mu_law = True
    else:
        model = st.models.SpecEncDec(ft_size=args.chunk)

    # In my experiments, DataParallel runs no faster than single-GPU for the same
    #   batch size.  It will allow you to run with larger batches, but...that's it.
    if args.parallel and (torch.cuda.device_count() > 1):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    #----------------------------------------------------------------
    #  Set additional training specs based on which model was chosen
    #----------------------------------------------------------------
    losslogger = st.utils.LossLogger(name=args.name)

    if (args.model != 'wavenet'):
        criteria = [torch.nn.MSELoss()]
        optimizer = torch.optim.Adam( model.parameters(), lr=args.lr, eps=5e-6)    #,  amsgrad=True)
    else:
        raise ValueError("Sorry, model 'wavenet' isn't ready yet.")
        criteria = [torch.nn.CrossEntropyLoss()]
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
    # Set up Training and Validation datasets
    #------------------------------------------
    print("Peparing Training data...")
    dataset = None#  st.audio.AudioDataset(args.length, chunk_size=args.chunk, effect=args.effect)
    X_train, Y_train = st.audio.gen_audio(sig_length, batch_size, device, chunk_size=args.chunk, effect=args.effect, mu_law=mu_law)
    print("Peparing Validation data...")
    X_val, Y_val = st.audio.gen_audio(sig_length, int(batch_size/5), device, chunk_size=args.chunk, effect=args.effect, mu_law=mu_law, x_grad=False)

    #--------------------
    # Call training Loop
    #--------------------
    model = train_model(model, optimizer, criteria, lambdas, X_train, Y_train, args, device,
        X_val=X_val, Y_val=Y_val, start_epoch=start_epoch, losslogger=losslogger, fs=fs,
        retain_graph=retain_graph, mu_law=mu_law, dataset=dataset)

    #--------------------------------
    # Evaluate model on Test dataset
    #--------------------------------
    eval_model(model, criteria, lambdas, losslogger, sig_length, chunk_size, device, args.effect,
        fs=fs, effect=args.effect, mu_law=mu_law, dataset=dataset)
    return

if __name__ == '__main__':

    main()

# EOF

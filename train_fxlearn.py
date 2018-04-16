#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis & S.H. Hawley'
__copyright__ = 'MacSeNet'

# imports
import numpy as np
import torch
import argparse
import os
import fxlearn as fxl    # This is where I'm putting things...

def train_model(model, optimizer, criterion, start_epoch = 0, max_epochs=100,
    batch_size=100, sig_length=8192, fs=44100.,
    tol=1e-14, change_every=10, save_every=100, plot_every=200):

    for epoch_iter in range(max_epochs - start_epoch):
        epoch = start_epoch + epoch_iter
        print('Epoch ', epoch,'/', max_epochs, sep="", end=":")

        # change the input dataset to encourage generality
        if (0 == epoch_iter % change_every):
            input_var, target_var = fxl.utils.make_signals(sig_length, fs, batch_size=batch_size)   # generate numpy arrays

        optimizer.zero_grad()
        wave_form = model(input_var)   # run the neural network forward
        loss = criterion(wave_form, target_var)

        loss_val = loss.data.cpu().numpy()[0]
        fxl.utils.print_loss(loss_val)

        if (epoch > start_epoch):
            if (0 == epoch % save_every) or (epoch >= max_epochs-1):
                print("Saving checkpoint...")
                fxl.utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), #'best_prec1': loss_val,
                    'optimizer' : optimizer.state_dict(),}, False)
            if (0 == epoch % plot_every):
                outfile = 'progress.pdf'
                print("Saving progress report to ",outfile,": ",end="")
                fxl.utils.make_report(input_var, target_var, wave_form, outfile=outfile, epoch=epoch)

        if loss.data.cpu().numpy() < tol:
            break

        loss.backward(retain_graph=True)      # retain_graph=True needed for RNNs, it seems
        optimizer.step()

    return model



# One additional model evaluation on new 'test; data
def eval_model(model, sig_length, fs):
    input_var, target_var = fxl.utils.make_signals(sig_length, fs, num_waves=4)
    wave_form = model(input_var)    # run network forward
    outfile = 'final.pdf'
    print("Saving final Test evaluation report to",outfile,": ",end="")
    fxl.utils.make_report(input_var, target_var, wave_form, outfile=outfile)
    return



def main():
    if torch.has_cudnn:
        print('Running on CUDA')
    else:
        print('No CUDA')

    # useful for ensuring similar initializations when testing different models
    torch.manual_seed(1)
    np.random.seed(1)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='fxlearn')
    parser.add_argument('--epochs', default=10000, type=int, help="Number of iterations to train for")
    parser.add_argument('--length', default=8192, type=int, help="Length of audio signals")
    parser.add_argument('--fs', default=44100, type=int, help="Sample rate in Hertz")
    parser.add_argument('--change', default=10, type=int, help="Changed data every this many epochs")

    parser.add_argument('--model', default='spectral', type=str,
                    help="Model type: 'spectral' (default) or 'seq2seq'")
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    args = parser.parse_args()
    print("args = ",args)
    sig_length = args.length
    fs = args.fs

    # Set up model and training criteria
    if ('seq2seq' == args.model):
        model = fxl.models.Seq2Seq()
    else:
        model = fxl.models.SpecEncDec()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([ {'params': model.parameters()}], lr = 0.0003,  amsgrad=True)#, weight_decay=0.01)



    if os.path.isfile(args.resume):     # checkpoint recovery
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        #best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        start_epoch = 0


    if torch.has_cudnn:
        model.cuda()

    # Training Loop
    model = train_model(model, optimizer, criterion, start_epoch=start_epoch, max_epochs=args.epochs,
        fs=fs, sig_length=sig_length, change_every=args.change)

    # Show performance on test data:
    eval_model(model, sig_length, fs)
    return

if __name__ == '__main__':

    main()

# EOF

![images/stlogo.png](images/stlogo.png)
# SignalTrain

Learning audio effects with neural networks

### Installation:
`pip install -r requirements.txt`

### Running:
`python train_signaltrain.py` or `./train_signaltrain.py`

(Run with `--help` for list of command-line options.)

Progress updates will be written to `progress0*` files, including `progress0.pdf`, which may look like this:
![progress_example.png](images/progress_example.png)

### Directory Structure:
```
signaltrain/    The main library
    +--- models.py      Where the pytorch classes & associated routines are
    +--- audio.py       Utilities related to audio production, effects, & more
    +--- utils.py       Grab bag of other utilities
docs/                   Not much here at the moment
experiments/    Where I do various runs and report on them
```

### TODO:
* [x]  Make checkpoint of 'id' (Autoencoder) 'effect' to use for (optionally) initializing network.
* [ ]  More effects! Leon Fedden's [RenderMan](https://github.com/fedden/RenderMan) VST host was just updated to Python 3.6.  I have yet to try it out. (A Python VST host is something I've wanted for this project for a long time!)
* [ ]  Integrate TensorBoard, e.g. via Pytorch [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
* [ ]  Write a Dataset generator class, e.g. via pytorch's [torchaudio](https://github.com/pytorch/audio)
* [ ]  Try mu-law companding to see if it improves SNR
* [ ]  Add parameterized controls, e.g. by concatenating in the middle of the network. 
* [ ]  A wavenet architecture? 
* [ ]  Try replacing MSE loss with an audio classifier and turn this into a GAN?  (Note: tried [a simple version of this idea](https://gist.github.com/drscotthawley/f0ecdc49d1c98d20dae26eb115b044b8) which didn't work out. Maybe I'm doing it wrong?) 


#### History:
Previously this repo was named "fxlearn".  Revised name "SignalTrain" suggested by [Rex Paul Schnelle](https://rexmusic.us/).

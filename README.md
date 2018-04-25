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
- More effects! Leon Fedden's [RenderMan](https://github.com/fedden/RenderMan) VST host was just updated to Python 3.6.  I have yet to try it out. (A Python VST host is something I've wanted for this project for a long time!)
- Try mu-law companding to see if it improves SNR
- A wavenet architecture? 


#### History:
Previously this repo was named "fxlearn".  Revised name "SignalTrain" suggested by [Rex Paul Schnelle](https://rexmusic.us/).

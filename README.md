# SignalTrain
Learning audio effects with neural networks

Authors: Scott Hawley, Stylianos Mimilakis, with some code by Ben Colburn


### Requirements:

    - [PyTorch](https://pytorch.org/) v1
    - Python 3.6 or greater
    - Recommended: CUDA 9.0 or greater

### Installation:

The simplest (& recommended) way to install is via [Anaconda](https://www.anaconda.com/) using the `freeze.yml` file:

    conda env create -f freeze.yml

Or you can try the `requirements.txt` file with pip:

    pip install -r requirements.txt


### Running

    cd main_scripts
    ./gen_synth_data.py
    ./train_with_knobs.py

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

If you run into trouble installing, try installing individually the packages in requirements.txt, e.g.

    conda install -c conda-forge librosa

...and/or, create a brand-new `conda` environment start from there. On Google Cloud Compute, I had to do the following

    conda create --name signaltrain python=3.6 
    conda activate signaltrain
    pip install -r requirements.txt 

### Running

    ./run_train.py --help 

will display a list of options

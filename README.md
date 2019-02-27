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

This will create an environment called 'signaltrain' which you can enable via `conda activate signaltrain`.

Alternatively, you can try pip and the `requirements.txt` file:

    pip install -r requirements.txt

If you run into trouble installing, try installing individually the packages in requirements.txt, e.g.

    conda install -c conda-forge librosa

...and/or, create a clean `conda` environment start from there. On Google Cloud Compute, I had to do the following

    conda create --name signaltrain python=3.6 
    conda activate signaltrain
    pip install -r requirements.txt 

### Dataset creation

For synthetic data using using the 4-knob compressor routine, this will create a directory called 'mydata' with Train/ and Val/ directories, and synthesize 5-second-long audio files, for 10 settings *per knob*: 

    ./gen_dataset.py mydata --dur 5  --effect comp_4c --sp 10 
    
To incorporate (or add) other audio files (e.g. music) to this, specify `--inpath` pointing to a directory which should *already contain* Train/ and Val/ (and maybe Test) directories:

    ./gen_dataset.py mydata --dur 5  --inpath ~/datasets/signaltrain/music/  --effect comp_4c --sp 10  



### Training

    ./run_train.py --help 

will display a list of options

### Contents

```
├── README.md
├── LICENSE
├── run_train.py              # main script for training
├── freeze.yml                # frozen conda environment
├── requirements.txt          # for pip install
├── signaltrain               # main lib
│   ├── train.py              # main training routine
│   ├── audio.py              # 'most' of the audio and plugin-related routines
│   ├── data.py               # AudioDataset routines
│   ├── __init__.py
│   ├── io_methods.py         # status messages, and some unused audio routines
│   ├── learningrate.py       # implentation of fast.ai learning rate scheduling
│   ├── loss_functions.py     # this is its own file only because one of us made it so ;-) 
│   ├── misc.py               # cosmetics
│   └── nn_modules            # NN architecture routines: 'fourier' transforms, autoencoder
│       ├── cls_fe_dct_bases.py
│       ├── cls_fe_dft.py
│       ├── __init__.py
│       └── nn_proc.py
├── utils
│   ├── gen_synth_data.py    # generates file dataset of synthetic data
│   └── reshuffle_testval.py
└── demo                     # Jupyter notebook with sliders
    ├── Leadfoot_ScottHawley.wav
    ├── modelcheckpoint_4c.tar
    ├── modelcheckpoint_denoise.tar
    └── SliderDemo.ipynb
```

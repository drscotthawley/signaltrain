```
 ~.~.~.~.      
 ____    `.    
 ]DD|_n_n_][   
 |__|_______)  
 'oo OOOO oo\_ 
~+~+~+~+~+~+~+~
```
# SignalTrain
Learning time-dependent nonlinear audio effects with neural networks

Code Authors: Scott Hawley, Stylianos Mimilakis, with some code by Ben Colburn

**Demo Page:** [signaltrain.herokuapp.com/](signaltrain.herokuapp.com/)

Other demo options: [Jupyter Notebook](https://github.com/drscotthawley/signaltrain/blob/master/demo/SliderDemo.ipynb), [Colab Notebook](https://colab.research.google.com/drive/1ZIij0CqfISDrgb3XclMrU-OILFDpQEJ0) (but the Colab runs very slow slow) 

Paper preprint at [https://arxiv.org/abs/1905.11928](https://arxiv.org/abs/1905.11928), slightly revised (& slightly shorter) [version accepted as a "full paper"](http://hedges.belmont.edu/~shawley/signaltrain_paper_aes.pdf) for [AES 147](http://www.aes.org/events/147/). Paper authors are Scott H. Hawley, Benjamin Colburn, and Stylianos I. Mimilakis. Title is "Profiling Audio Compressors with Deep Neural Networks".

**Clarification:** As we say in the paper, the code is written in pursuit of the goal of learning *general* audio effects, not just compressors. If you only want to do compressors, our method is 'overkill'.  But for the paper, we focused on compressors because they're both "hard" and "of practical interest". Further demonstrations of other effects are in progress. 


*Disclaimer: This is a 'research' code as apposed to a general utility package. Currently the level of this release is one of "openness" for reproducibility. It is not yet on the level of a fully-maintained package.  But feel free to send Issues & PR's and I'll try to accomodate them!*

### Main Requirements:

- Python 3.8 or greater
- [PyTorch](https://pytorch.org/) v1.7.1
- **Strongly recommended**: CUDA 9.0 or greater. (This code can work without CUDA but it would be slow and the installation procedure would need to follow Method 2 below.)

see [requirements.txt](requirements.txt) for more. 

### Installation:

First clone this repository: 

    git clone git@github.com:drscotthawley/signaltrain.git
    
Then cd into the directory

	cd signaltrain

Start from a clean environment

	conda create --name signaltrain python=3.8

Install suitable version of pytorch for your environment, e.g. in linux

	conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
	
Then try running the program and install whatever it needs.

Current not ready for the following methods.

>After that, choose one of the following two methods: 

>#### Method 1.
>The simplest (& recommended) way to install is via [Anaconda](https://www.anaconda.com/) using the `freeze.yml` file:

>    `conda env create -f freeze.yml`
    
>for windows users

>    `conda env create -f environment.yml`

>This will create an environment called 'signaltrain' which you can enable via `conda activate signaltrain`.

>Notes on this method: 

>- if you want to give the environment a different name, you can run `conda env create -f freeze.yml --name <your_env_name>`)
>- this method will pull a particular version of CUDA.  If you already have CUDA installed, you may get an error like `RuntimeError: cuda runtime error (11) : invalid argument at /opt/conda/conda-bld/pytorch_1544174967633/work/aten/src/THC/THCGeneral.cpp:405`.  This is common when one mixes versions of CUDA.  You may wish to try Method 2.

>#### Method 2. 
>Alternatively, or if Method 1 fails, you can try pip and the `requirements.txt` file:

>    `pip install -r requirements.txt`

>If you run into trouble installing, try installing individually the packages in requirements.txt, e.g.

>    `conda install -c conda-forge librosa`

>...and/or, create a clean `conda` environment start from there. On Google Cloud Compute, I had to do the following

>    `conda create --name signaltrain python=3.8
>    conda activate signaltrain
>    pip install -r requirements.txt`

### Datasets

#### Dataset creation
This is easy and fast. 

Although training program `./run_train.py` *can* generate its own data on the fly, but you'll find it faster to try out this repo if you pre-generate a bunch of data.

For synthetic data using using the 4-knob compressor routine, the following will create a directory called 'mydata' with Train/ and Val/ directories, and synthesize 5-second-long audio files, for 10 settings *per knob*:

    ./gen_dataset.py mydata --dur 5  --effect comp_4c --sp 10
    
(This runs in parallel using however many CPUs you have, so it's pretty fast.)

To incorporate (or add) other audio files (e.g. music) to this, specify `--inpath` pointing to a directory which should *already contain* Train/ and Val/ (and maybe Test) directories:

    ./gen_dataset.py mydata --dur 5  --inpath ~/datasets/signaltrain/music/  --effect comp_4c --sp 10  

#### Pre-existing datasets

[SignalTrain LA2A Dataset](https://zenodo.org/record/3348083) (21.0 GB) ![doi_image](https://zenodo.org/badge/DOI/10.5281/zenodo.3348083.svg) by Benjamin Colburn & Scott H. Hawley


### Training

To train, one runs the file `./run_train.py`, which will run in a standalone mode with no options using data it generates on-the-fly, however if you just ran one (or both) of the `gen_dataset.py` lines above, then it will run faster if you tell the program to load that data:

    ./run_train.py --path mydata

For other variations, running

    ./run_train.py --help

will display a list of options, e.g., number of epochs, model size, batch size, etc.

Advisories re. options:

- If your GPU can has a native FP16 representation, you may wish to run with the option `--apex O2` which will do Mixed Precision training (which will run faster and use less memory).
- The default number of epochs is 1000, which with other defaults will take 10.6 hours on a dedicated RTX 2080Ti GPU running in Mixed Precision.  You might want to set `--epochs 100` at first, or choose an even smaller number of epochs. 
- **Working on it:** with the `--checkpoint <checkpoint.tar>` option, one *should* be able to start from earlier model checkpoint files (i.e., in order to use pre-trained weights), however changes in the model structure over time mean that some of the checkpoint files in this repo made from earlier model architectures are incompatible with the new model structure, resulting in conflicts.  We're working to update these files.

### Using trained model on audio files

Copy `predict_long.py` to the root directory of the program, then

	./predict_long.py modelcheckpoint.tar TEST_AUDIO.wav --effect EFFECT_NAME --knobs='a,b,c,d' 
	
Change `TEST_AUDIO.wav` to the name of your audio file, `EFFECT_NAME` is the name of the effect, and the string in the '' in  `--knobs='a,b,c,d' ` should be change to your desired knob settings.

### Contents

```
├── README.md
├── LICENSE
├── run_train.py              # main script for training
├── freeze.yml                # frozen conda environment
├── requirements.txt          # for pip install
├── index.php                 # only used when deploying web demo / heroku app
├── Procfile                  # only used when deploying web demo / heroku app
├── signaltrain               # main lib
│   ├── train.py              # main training routine
│   ├── audio.py              # 'most' of the audio and plugin-related routines
│   ├── data.py               # AudioDataset routines
│   ├── nn_proc.py            # neural network architecture(s)
│   ├── cls_fe_dft.py         # convnet trainable STFT routines
│   ├── cls_fe_dct_bases.py   # utilities used by previous file
│   ├── io_methods.py         # status messages, and some unused audio routines
│   ├── learningrate.py       # implentation of fast.ai learning rate scheduling
│   ├── loss_functions.py     # this is its own file only because one of us made it so ;-)
│   └── misc.py               # cosmetics, loading/saving model files, logging
├── utils
│   ├── gen_synth_data.py    # generates file dataset of synthetic data
|   ├── lr_finder.py         # learning rate finder app (reproduces work by Fast.AI)
|   ├── predict_long.py      # applies an audio effect & model to a long audio file
|   ├── resample_dataset.py  # used for testing only; apply new sample rate to entire dataset
│   └── reshuffle_testval.py # reshuffles test/val datasets.  (optional; shoudn't need to use this)
└── demo                     # Demos with sliders
    ├── SliderDemo.ipynb     # Jupyter notebook of slider demo
    ├── modelcheckpoint_4c.tar        # 4-control compressor model file
    ├── modelcheckpoint_denoise.tar   # another model file for denoising example
    ├── Leadfoot_ScottHawley_clip.wav     # sample audio file
    ├── index.html           # used only for web-based demo app (on heroku)
    ├── bokeh_sliders.py     # used only for web-based demo app
    ├── model_comp4c_4k.tar  # model checkpoint used for web-based demo app
    └── model_graph.svg      # model graph, shown on web web demo page
```

**Other remarks / Opportunities for Pull Requests!:** 

1. Transfer learning? Yea, work in progress.  In theory a lot of the represenation/weights can carry over from one effect to the next, I just haven't written the code to copy the weights appropriately. If you want to add this, I'll take a PR! :-) 

2. Memory: If you're not generating 'on the fly,' then this thing is a 'memory hog' in the sense that it loads the whole dataset at the beginning -- this is done for the sake of *speed*, because in my experience disk I/O slows things down more than I find acceptable, and my machines have ample RAM (either [64 GB](https://pcpartpicker.com/user/drscotthawley/saved/bbw8dC) or [128 GB](https://pcpartpicker.com/b/j7J8TW)).  If you have a 'fast' way of reducing memory load, feel free to send a PR! 

3. Mixed Precision (MP): As we note in the paper, the use of MP does not increase the error noticeably *even though* we're generating audio as raw floats. This may be because it's swamped by other error. 

4. What about parallelism?  In my experience, on my machines, running 'data parallel' on multiple GPUs only produces a small speedup (i.e. not worth it, better to run multiple triviall-parallel jobs learning different things). 


### Citing This Work
There are 'two versions' of the original SignalTrain paper. The original on arXiv was longer, but the 'official' version was a slightly shorter Category 1 paper at the 2019 fall AES conference.  Since the latter is the peer-reviewed version, we prefer that you cite that:
```
@conference{signaltrain,
		title = {Profiling Audio Compressors with Deep Neural Networks},
		author = {Hawley, Scott and Colburn, Benjamin and Mimilakis, Stylianos Ioannis},
		booktitle = {Audio Engineering Society Convention 147},
		month = {Oct},
		year = {2019},
		url = {http://www.aes.org/e-lib/browse.cfm?elib=20595}
		}
```

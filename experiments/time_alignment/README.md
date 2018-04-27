
## Basic Overview
First attempt: with multiple, randomly-placed 'plucks'.  Red is "on the grid".
![image1](../../images/TimeAlignPic1_many_plusminus.png)


That seemed hard, so decided to focus on just one event, which 'should' be in the middle of window:

![image](../../images/TimeAlignPic2_single_random_silence.png)


Ok, so that worked, so what about if it's not preceded by silence?  Here's noise:

![image](../../images/TimeAlignPic2_single_random_noisered.png)

...interesting that it doesn't learn the noise.  That's ok.


In reality, though, there will be a precding event.  In this case, interestingly, when the random event occurs too early, the newtork learns to **generate more of the preceding signal** to fill in the gap:

![image](../../images/TimeAlign_generatemissing.png)

![image](../../images/TimeAlign_generatemissing2.png)	

![image](../../images/TimeAlignanotherexample.png)

Obviously this is a function of how it was trained.  If it were trained on a dataset that did cross-fades, then presumably (I have yet to try this) it would learn to approximate a cross-fade.

Current maximum length I can get is 10500 samples, or around 340ms at 44.1Hz.  This is sufficient for most drum samples.  


## Skip connections:

Added skip connections to the model; seems to have reached the same level of error as before, only gets there faster:
![image](loss_skips.png)


## Adding `strength` 'knob'

Instead of moving the erroneous event exactly 'onto the grid' (henceforth "OTG"), I introduced a tunable parameter `strength` for which
`strength=0` means leaving the event unedited, and `strength=1` means put it exactly on the grid.  Mathematically,
given the onset time of the input event, t_i, and the onset time of the fully-edited OTG version t_g,
we define the target time to be

`t_target = strength * t_g + (1 - strength) * t_i`

...i.e., strength linearly interpolates between the input time and the OTG time.  

**Results**: With the same learning rate as before, this is hard to do.  But if you reduce the learning rate, you can get some nice results:

![vloss graph](strength=0.5/vloss_hist_graph_strength=0.5.png)

![progress graph](strength=0.5/progress0.png)


## Future work:

- Pass the strength parameter into the neural network model, e.g. concatenate it in the 'bottleneck' part of the network 
- Read in actual audio samples of drums
- For drum samples, having a pytorch Dataset class (e.g. via torchaudio) might be a good idea. 

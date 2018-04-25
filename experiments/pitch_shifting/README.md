So far...

Given a set of N random amplitudes A_i {i=1..N} and frequencies f_i, generate signals

![img](https://latex.codecogs.com/gif.latex?X=\sum_{i=1}^{N}A_{i}\sin(f_i&space;t))

and a corresponding set of pitch-shifted and amplitude-shifted (and phase-shifted!) signals

![img](https://latex.codecogs.com/gif.latex?Y=\sum_{i=1}^{N}c_1&space;A_{i}\cos(c_2&space;f_i&space;t))

where c_1 and c_2 are constant coefficients, e.g. c_1= 0.37 and c_2 = 0.43,

...the system is able to learn the mapping X->Y for sets of X it hasn't seen before. 

But this is not true pitch-shifting.  I'd hope to make use of VST host [Renderman](https://github.com/fedden/RenderMan) if it ever gets updated from Python 2.7 to 3.5.

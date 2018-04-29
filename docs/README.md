
Here we have, not proper documentation yet, just a list of notes as I
develop the system.

Here's a picture of the 'Spectral Shrink-Grow' model, where information flows
from top to bottom, and all layers are 'dense'/'Linear', all activations are
LeakyReLU (tried ELU & SELU, this worked best).

Parts in cyan are transformations in the time domain, parts in purple are in the
spectral domain, and the green FNN routines convert between the two. 

![model_diagram](model_diagram.png)

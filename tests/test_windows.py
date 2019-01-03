import numpy as np
from helpers import audio

chunk_size = 4096
batch_size= 200
overlap = 2048
signal_length = int( (chunk_size-overlap)*batch_size + overlap)
print("signal_length = ",signal_length)
x = np.arange(signal_length)
print("x.shape = ",x.shape,"x =\n",x)

print("overlap =",overlap)
newx = audio.sliding_window(x, chunk_size, overlap=overlap)
print("newx.shape = ",newx.shape,", newx =\n",newx)
print("total newx size = ",np.product(newx.shape))

y = audio.undo_sliding_window(newx, overlap)
print("y.shape =",y.shape,", y = \n",y)

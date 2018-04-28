 
 A compressor affect has not been possible at any time in the history of this project -- the LSTM networks simply 'refused' to learn anything (at least not on a discernable timescale).  
 
 Recently, however, adding skip connections to the shrink-grow FNN spectral model was able to produce something roughly matching the compressor:
 
 ![comp_image](../../images/progress0_comp_skips.png) 
 
 Notes on this graph:
 
 - The blip in the loss history is where the run was interrupted & restarted.
 - Each epoch takes about 1 second to execute. This gives you a rough idea of the time involved in training. 
 
 More work is needed. 
 
 
 ---
 
 (Note: I say "compressor" instead of "compression", only to distinguish this dynamic-range compression from "data compression".)
 

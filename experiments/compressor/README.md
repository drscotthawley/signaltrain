 
 A compressor affect has not been possible at any time in the history of this project -- the LSTM networks simply 'refused' to learn anything (at least not on a discernable timescale).  
 
 Recently, however, adding skip connections to the shrink-grow FNN spectral model was able to produce something roughly matching the compressor:
 
 ![comp_image](../../images/progress0_comp_skips.png) 
 
 The blip in the loss history is where the run was interrupted & restarted.
 
 More work is needed. 
 
 
 ---
 
 (Note: I say "compressor" instead of "compression", only to distinguish this dynamic-range compression from "data compression".)
 

#!/bin/sh

# just reads & spits out all wav files in current directory; makes scipy.io.wavfile "like" the files
for i in *.wav ; do
    sox "$i"  $(basename "${i/.wav}")q.wav  trim 0 900; mv $(basename "${i/.wav}")q.wav "$i"
done

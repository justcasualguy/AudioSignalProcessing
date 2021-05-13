import os

# import librosa.display
import scipy
import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import mfcc
import plots
import librosa
def getWavPath(oafOrYaf,emotion,word):
    rootPath = "TESS Speech Data"
    fullpath=os.path.join(rootPath,str.upper(oafOrYaf)+"_"+emotion,str.upper(oafOrYaf)+"_"+word+"_"+emotion+".wav")
    return fullpath
# x = np.array([[1], [4]], np.int32)
frame_length=0.025
frame_hop=0.01
num_mfccs=12
sample_rate, signal = scipy.io.wavfile.read(getWavPath("oaf","angry","back"))
frame_size= int(round(frame_length * sample_rate)) #in samples
signal = mfcc.pre_emphasis(signal,0.95)
frames = mfcc.framing(signal,sample_rate,frame_length,frame_hop)
frames_windowed = mfcc.hamming_window(frames, int(frame_length * sample_rate))
frames_magnitude = mfcc.dft(frames_windowed)
max= np.max(frames_magnitude)
linspace = np.linspace(0,sample_rate/2,len(frames_magnitude[0]))
frames_filtered=mfcc.mel_filter_banks(40,sample_rate,frames_magnitude)
coefficients = mfcc.dct(frames_filtered)[:,1:num_mfccs+1]
plots.plot_spectrogram(coefficients,signal,sample_rate)
mfcc = librosa.feature.mfcc(frames_filtered,sample_rate)[1:13]
plots.plot_spectrogram(mfcc,signal,sample_rate)
len(coefficients)




# signal_linespace=np.linspace(0,sample_rate,frames_magnitude[0])
# mfcc.mel_filter_banks(40,sample_rate/2)




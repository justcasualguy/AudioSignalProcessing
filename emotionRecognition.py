import os

import librosa.display
import scipy
import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import mfcc
def getWavPath(oafOrYaf,emotion,word):
    rootPath = "TESS Speech Data"
    fullpath=os.path.join(rootPath,str.upper(oafOrYaf)+"_"+emotion,str.upper(oafOrYaf)+"_"+word+"_"+emotion+".wav")
    return fullpath
# x = np.array([[1], [4]], np.int32)
frame_length=0.025
frame_hop=0.01
sample_rate, signal = scipy.io.wavfile.read(getWavPath("oaf","angry","back"))
signal = mfcc.pre_emphasis(signal,0.95)
frames = mfcc.framing(signal,sample_rate,frame_length,frame_hop)
frames_windows = mfcc.hamming_window(frames,frame_length)
# signal_linespace=np.linspace(0,signal.shape[0]/sample_rate,signal.shape[0])
# plt.plot(signal_linespace,signal)
# plt.title("Fraza \"Say the word back\", emocja: złość")
# plt.xlabel("Czas [s]")
# plt.ylabel("Amplituda")
# plt.show()



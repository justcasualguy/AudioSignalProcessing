import os

# import librosa.display
import scipy
import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import mfcc
import plots
import librosa
import Data

frame_length=0.025
frame_hop=0.01
num_filterbanks=40
num_mfccs=12

happyList=Data.FindData.findAudioFilesByEmotion(Data.Emotion.HAPPY)
sadList=Data.FindData.findAudioFilesByEmotion(Data.Emotion.SAD)[1:10]
sadList = list(map(lambda file : file.computeMfccs(frame_length,frame_hop,num_filterbanks,num_mfccs),sadList[1:10]))
for list in sadList:
    print(list.mfccs)









# signal_linespace=np.linspace(0,sample_rate,frames_magnitude[0])
# mfcc.mel_filter_banks(40,sample_rate/2)




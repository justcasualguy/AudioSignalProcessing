import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import matplotlib as mtp
from enum import Enum
import numpy as np
import os
def getWavPath(oafOrYaf,emotion,word):
    rootPath = "TESS Speech Data"
    fullpath=os.path.join(rootPath,str.upper(oafOrYaf)+"_"+emotion,str.upper(oafOrYaf)+"_"+word+"_"+emotion+".wav")
    return fullpath
def plot_magnitude_spectrum(signal,title,sr,f_ratio=1):
    ft=np.fft.fft(signal)
    magnitude_spectrum=np.abs(ft)

    plt.figure(figsize=(18,5))
    frequency=np.linspace(0,sr,len(magnitude_spectrum)) #create X axis( frequency bins)
    num_frequency_bins=int(len(frequency)*f_ratio)
    plt.plot(frequency[:num_frequency_bins],magnitude_spectrum[:num_frequency_bins])
    plt.xlabel("Frequency (Hz)")
    plt.title(title)
    plt.show()
def plot_spectrogram(Y,sr,hop_length,ax,y_axis="linear"):

    spectrogram=librosa.display.specshow(Y,sr=sr,
                                         cmap="coolwarm",
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis,
                                ax=ax)

    return spectrogram

def calcSpectrogram(audioFile,frame_size,hop_size):
    stft=librosa.stft(audioFile,n_fft=frame_size,hop_length=hop_size)
    spectrogram = np.abs(stft)**2
    return librosa.power_to_db(spectrogram)

angry_back, angry_back_sr =librosa.load(getWavPath("oaf","angry","back"))
happy_back,happy_back_sr=librosa.load(getWavPath("oaf","happy","back"))
fear_back,fear_back_sr = librosa.load(getWavPath("oaf","fear","back"))


# plot_magnitude_spectrum(angry_back,"oaf angry back",angry_back_sr,0.5)
# plot_magnitude_spectrum(happy_back,"oaf happy back",happy_back_sr,0.5)
FRAME_SIZE=2048
HOP_SIZE=512

# S_angry_back=librosa.stft(angry_back,n_fft=FRAME_SIZE,hop_length=HOP_SIZE)
# Y_angry_back=np.abs(S_angry_back)**2
# Y_angry_back_log=librosa.power_to_db(Y_angry_back)
# plot_spectrogram(Y_angry_back_log,angry_back_sr,HOP_SIZE,y_axis="log")

# fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False,figsize=(15,10))
# ax[0][0].set(title='ANGRY')
# ax[0][1].set(title='FEAR')
# ax[1][0].set(title='HAPPY')
# angry=plot_spectrogram(calcSpectrogram(angry_back,FRAME_SIZE,HOP_SIZE),angry_back_sr,HOP_SIZE,ax[0][0],"log")
# fear=plot_spectrogram(calcSpectrogram(fear_back,FRAME_SIZE,HOP_SIZE),angry_back_sr,HOP_SIZE,ax[0][1],"log")
# happy=plot_spectrogram(calcSpectrogram(happy_back,FRAME_SIZE,HOP_SIZE),angry_back_sr,HOP_SIZE,ax[1][0],"log")
# plt.subplots_adjust(hspace=1)
# format="%.2f"
# fig.colorbar(angry,ax=[ax[0][0]],format=format)
# fig.colorbar(fear,ax=[ax[0][1]],format=format)
# fig.colorbar(angry,ax=[ax[1][0]],format=format)
#
#
# plt.show()
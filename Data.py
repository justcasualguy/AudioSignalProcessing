import os.path as path

from enum import Enum
from scipy.io.wavfile import read as read
import os
import mfcc
class Actor(Enum):
    OLD_FEMALE="OAF"
    YOUNG_FEMALE="YAF"
class Emotion(Enum):
    ANGRY = "angry"
    DISGUST="disgust"
    FEAR="fear"
    HAPPY="happy"
    NEUTRAL="neutral"
    PLEASANT_SURPRISED="pleasant surprised"
    SAD="sad"

class FindData:
    rootPath = "TESS Speech Data"
    def getWavPath(actor: Actor, emotion:Emotion, word):
        emotionPath=emotion.value.replace(" ","_")
        emotionName = emotion.value
        if(emotion==Emotion.PLEASANT_SURPRISED):
            emotionName = "ps"
        fullpath = path.join(FindData.rootPath, actor.value + "_" + emotionPath,
                             actor.value + "_" + word + "_" + emotionName + ".wav")
        return fullpath

    def findAudioFilesByEmotion(emotion: Emotion):
        emotionPath = emotion.value.replace(" ", "_")
        files = []
        for folderName in os.listdir(FindData.rootPath):
            if emotionPath in folderName:
                for file in os.listdir(os.path.join(FindData.rootPath, folderName)):
                    actor, word, wav = file.split("_")
                    files.append(AudioFile(emotion, word, actor))
        return files


class AudioFile:

    def __init__(self,emotion,word,actor: Actor):
        self.emotion=emotion
        self.word=word
        if actor == "OAF":
            actor=Actor.OLD_FEMALE
        else:
            actor=Actor.YOUNG_FEMALE
        self.actor = actor
        self.sample_rate,self.signal=read(
            FindData.getWavPath(self.actor,self.emotion,self.word)
        )
        self.length = round(self.signal.shape[0]/self.sample_rate,2)

    def computeMfccs(self,frame_length_sec,frame_hop_sec,num_filterbanks,num_mfccs):
        self.mfccs = mfcc.getMfccs(self.signal,self.sample_rate,frame_length_sec,frame_hop_sec,num_filterbanks,num_mfccs)
        return self


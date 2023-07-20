from moviepy import editor
from pydub import AudioSegment
from os import path
from os import system
from tkinter.filedialog import *
import librosa
import numpy as np
import keras
system('cls')
vid = askopenfilename()
data, sr = librosa.load(vid)
down_d = librosa.resample(data, orig_sr = sr,target_sr=16000)
down_d = librosa.util.fix_length(down_d, size=5051696)
data = librosa.feature.mfcc(y =  down_d,sr = sr,n_mfcc =40)
data = (data-np.mean(data))/np.std(data)
data = np.array(data)
# data = np.concatenate((data,data),axis=0)
data = np.expand_dims(data, axis = 0)
reconstructed_model = keras.models.load_model("Audio95.h5")
pred = reconstructed_model.predict(data)
pred = np.argmax(pred,axis = 1).tolist()[0]
dict = {
    1: "UnderConfident",
    2: "Less Confident",
    3: "Confident",
    4: "Very Confident",
    5: "Over Confident"
}
print("Audio Confidence Score is ",dict[pred+1])
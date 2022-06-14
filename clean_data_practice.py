#! .\bchlr-venv\scripts\python.exe

from math import ceil
import os
import glob
import librosa
import numpy as np
from statistics import mean
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
from python_speech_features import mfcc, logfbank

# Random copied visualization funktions
def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                            sharey=True, figsize=(16,4))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                            sharey=True, figsize=(16,4))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                            sharey=True, figsize=(16,4))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                            sharey=True, figsize=(16,4))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

# Calculate the fft
def calc_fft(y, rate):
    n = len(y) #calculate the window length for rfftfreq
    freq = np.fft.rfftfreq(n, d=1/rate) #d is the spacing between samples
    Y = abs(np.fft.rfft(y)/n) #/n for normalization
    return (Y, freq)

# Calculate an envelope
def envelope(y, rate, threshhold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean
    for mean in y_mean:
        if mean > threshhold:
            mask.append(True)
        else:
            mask.append(False)
    return mask



# LOADING DATA ----------------------------------------------------------------------------------------------------
# Create array with [[filename], [instrument], [length], [full path]]
glob = glob.glob('data\wavfiles\*\*.wav')
df = np.transpose([[os.path.basename(f) for f in glob], #get all files in all wavefile directorys
        [os.path.basename(os.path.dirname(f)) for f in glob], #get the names of all the corresponding directories
        [0 for f in glob], #add in an array of extra 0 to later fill with length values
        [f for f in glob]]) #get all paths
        
# Calculate the length in [[filename], [instrument], [length], [full path]]
# Also calculate sampling rate
sr = 0
for i in range(len(df)):
    rate, signal = wavfile.read('data/wavfiles/' + df[i, 1] + "/" + df[i, 0]) #read the wav file
    df[i, 2] = signal.shape[0]/rate #calculate the length of each sample and save it in the array
    sr = rate

# Calculate the distrubution for each instrument (mean leangth)
classes = list(np.unique(df[:, 1])) #get a list of all unique classes
class_dist = np.zeros_like(classes)
for i, c in enumerate(classes):
    class_list = np.where(df[:, 1] == c) #get all positions in the array with the same class c
    class_length = [l for l in df[class_list, 2]] #get the length of all samples with the same class
    class_length = [float(l) for l in class_length[0]] #char to float
    class_dist[i] = mean(class_length) #calculate the mean to get the class distribution

# Plot the class distribution data
fig, ax = plt.subplots()
ax.set_title('Class Distribution')
ax.pie(class_dist, labels=classes, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
plt.draw()

# Create dictionarys
signals = {}
fft = {}
fbank = {}
mfccs = {}

# Calculate values for plotting 1 example from each class
for i, c in enumerate(classes):
    wav_file = 'data/wavfiles/' + df[i*30, 1] + '/' + df[i*30, 0] #put together the full wav file path for 1 sample from each class
    signal, rate = librosa.load(wav_file, sr=sr) #load the wav file (ibrosa.load normalizes the data between (-1, 1))
    #mask = envelope(signal, rate, 0.0005) #get a mask to get rid of "dead space" in the signal
    #signal = signal[mask] #apply mask to sigan (this likely wont be neccesary for speech data)
    signals[c] = signal #add signal to dictionary with key c (this won't work with more than 1 sample for each class)
    fft[c] = calc_fft(signal, rate) #calc fft

    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=ceil(sr/40)).T #TODO look up wtf filterbank and mfcc is
    fbank[c] = bank
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=ceil(sr/40)).T #signal[:rate] returns 1 sec of the signal
    mfccs[c] = mel

# Plot the signals, fft, fbank, mfcc
plot_signals(signals)
plt.draw()
plot_fft(fft)
plt.draw()
plot_fbank(fbank)
plt.draw()
plot_mfccs(mfccs)
plt.draw()

if len(os.listdir('data/clean')) == 0:
    for f, path in zip(df[:, 0], df[:, 3]):
        signal, rate = librosa.load(path, sr=16000) #downsample because high frequencies aren't needed
        #mask = envelope(signal, rate, 0.0005) #TODO do all data preprocessing here
        wavfile.write(filename='data/clean/'+f, rate=rate, data=signal) #give the preprocessed signals here

plt.show()
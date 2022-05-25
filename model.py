#! .\bchlr-venv\scripts\python.exe

from asyncio.windows_events import NULL
import os
import glob
import numpy as np
from statistics import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
from python_speech_features import mfcc

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def build_rand_feat():
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in range(10): #n_samples
        rand_class = np.random.choice(classes, p=prob_dist)
        file = np.random.choice([sample for i, sample in enumerate(df[:, 0]) if df[i, 1] == rand_class])
        rate, wav, = wavfile.read('data/clean/' + file)
        rand_index = np.random.randint(0, wav.shape[0] - config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft).T
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample if config.mode == 'conv' else X_sample.T) #make sure the data has the format that the nn model needs
        y.append(classes.index(rand_class))
    X, y = np.array(X), np.array(y) #no sure if this is such a good idea
    X = (X - _min) / (_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = torch.from_numpy(y).type(torch.int64)
    y = F.one_hot(y, num_classes=10)
    print(y)
    return X, y

# TODO ?
class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate = 16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)

# Create array with [[filename], [instrument], [length]]
glob = glob.glob('data\wavfiles\*\*.wav')
df = np.transpose([[os.path.basename(f) for f in glob], #get all files in all wavefile directorys
        [os.path.basename(os.path.dirname(f)) for f in glob], #get the names of all the corresponding directories
        [0 for f in glob]]) #add in an array of extra 0 to later fill with length values

# Calculate the length in [[filename], [instrument], [length]]
# Also calculate sampling rate
sr = 0
for i in range(len(df)):
    rate, signal = wavfile.read('data/clean/' + df[i, 0]) #read the wav file
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

# TODO what are we doing here?
n_samples = 2 * int(df[:, 2].astype(float).sum()/0.1)
prob_dist = class_dist.astype(float) / class_dist.astype(float).sum()
choices = np.random.choice(classes, p=prob_dist)

# Plot the class distribution data
fig, ax = plt.subplots()
ax.set_title('Class Distribution')
ax.pie(class_dist, labels=classes, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
#plt.show()

# TODO ?
config = Config(mode='conv')

# TODO ?
if config.mode == 'conv':
    X, y = build_rand_feat()
elif config.mode == 'time':
    X, y = build_rand_feat()
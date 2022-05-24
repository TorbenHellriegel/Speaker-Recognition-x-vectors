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
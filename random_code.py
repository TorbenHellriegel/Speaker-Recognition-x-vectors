#! .\bchlr-venv\scripts\python.exe

from glob import glob
import os
import glob
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from scipy.io import wavfile
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from python_speech_features import mfcc

globs = glob.glob('data\wavfiles\*\*.wav')
for i, g in enumerate(globs):
    if i == 0:
        rate, sample = wavfile.read(g, np.dtype)

# rand_index = np.random.randint(0, sample.shape[0] - 1600)
# sample = sample[rand_index:rand_index+1600]

print(sample)
print(sample.shape)
print(rate)

x = mfcc(sample, rate, numcep=13, nfilt=26, nfft=512)
print(x)
print(x.shape)
#! .\bchlr-venv\scripts\python.exe

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile

# Push calculation to gpu if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LOAD DATA

# TODO adjust based on what the dataset looks like
class Dataset(Dataset): 
    def __init__(self):
        self.filenames = 0
        self.classes = 0

        self.samples = 0
        self.sampling_rates = 0
        self.n_samples = 0

    # Returns the sample at the given index with the predefined transform
    # Can be called as dataset[i]
    def __getitem__(self, index):
        return torch.from_numpy(self.samples[index]), self.classes[index]
    
    # Load the training data and save all relevant info in arrays
    # TODO PREPROCESS DATA BEFORE THIS
    def load_train_data(self):
        # Get the paths to all the data samples
        globs = glob.glob('data\wavfiles\*\*.wav')

        # Get only the testing data samples
        # TODO put training and testing data in different folders to make this step easier
        for i, g in enumerate(globs):
            if(i%2 == 0):
                globs[int(i/2)] = globs[i]
        globs = globs[:int(len(globs)/2)]

        # Get the filenames and class names from the paths
        self.filenames = np.array([os.path.basename(f) for f in globs])
        self.classes = np.array([os.path.basename(os.path.dirname(f)) for f in globs])

        # Gat the list of samples and the sampling rate
        samples = []
        rates = []
        for g in globs:
            rate, sample = wavfile.read(g, np.dtype)
            samples.append(np.array(sample))
            rates.append(rate)

        self.samples = samples
        self.sampling_rates = rates
        self.n_samples = len(self.samples)
    
    # Load the testing data and save all relevant info in arrays
    # TODO PREPROCESS DATA BEFORE THIS
    def load_test_data(self):
        # Get the paths to all the data samples
        globs = glob.glob('data\wavfiles\*\*.wav')

        # Get only the testing data samples
        # TODO put training and testing data in different folders to make this step easier
        for i, g in enumerate(globs):
            if(i%2 == 1):
                globs[int((i-1)/2)] = globs[i]
        globs = globs[:int(len(globs)/2)]

        # Get the filenames and class names from the paths
        self.filenames = np.array([os.path.basename(f) for f in globs])
        self.classes = np.array([os.path.basename(os.path.dirname(f)) for f in globs])

        # Gat the list of samples and the sampling rate
        samples = []
        rates = []
        for g in globs:
            rate, sample = wavfile.read(g, np.dtype)
            samples.append(np.array(sample))
            rates.append(rate)

        self.samples = samples
        self.sampling_rates = rates
        self.n_samples = len(self.samples)

train_dataset = Dataset()
test_dataset = Dataset()

train_dataset.load_train_data()
test_dataset.load_test_data()

for train, test in zip(train_dataset, test_dataset):
    print("train: ",  train)
    print("test: ",  test)
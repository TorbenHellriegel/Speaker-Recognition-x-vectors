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
    # Can be called as dataset[i] and should work with complex array manipulation
    def __getitem__(self, index):
        return torch.from_numpy(self.samples[index])

    # Returns the number of samples
    def __len__(self):
        return self.n_samples
    
    # Load the data and save all relevant info in arrays
    # TODO PREPROCESS DATA BEFORE THIS
    def load_data(self):
        # Get the paths to all the data samples
        globs = glob.glob('data\wavfiles\*\*.wav')

        # Get the filenames and class names from the paths
        self.filenames = np.array([os.path.basename(f) for f in globs])
        self.classes = np.array([os.path.basename(os.path.dirname(f)) for f in globs])

        # 
        samples = []
        rates = []
        for g in globs:
            rate, sample = wavfile.read(g, np.dtype)
            samples.append(np.array(sample))
            rates.append(rate)

        self.samples = samples
        self.sampling_rates = rates
        self.n_samples = len(self.samples)

class ToTensor:
    def __call__(self, sample):
        return torch.from_numpy(sample)

dataset = Dataset()

dataset.load_data()
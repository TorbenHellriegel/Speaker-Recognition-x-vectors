import glob
import math
import os

import numpy as np
import torch
from python_speech_features import mfcc
from scipy.io import wavfile
from torch.utils.data import Dataset


# TODO adjust based on what the dataset looks like
class Dataset(Dataset): 
    def __init__(self):
        self.classes = 0
        self.unique_classes = 0

        self.samples = 0
        self.sampling_rates = 0
        self.n_samples = 0

    # Returns the sample and class at the given index
    # Can be called as dataset[i] and works with dataloader
    def __getitem__(self, index):
        return torch.from_numpy(self.samples[index]), self.unique_classes.index(self.classes[index])

    def __len__(self):
        return self.n_samples
    
    # Load the training data and save all relevant info in arrays
    # TODO Preprocess data before this if neccessary
    def load_train_data(self, mfcc_numcep=24, mfcc_nfilt=26, mfcc_nfft=512):
        # Get the paths to all the data samples
        globs = glob.glob('../VoxCelebDownload/data/VoxCeleb/vox1_dev_wav/id1000*/*/*.wav') #TODO replace id100* with * to load all samples

        # Get the class names from the paths
        self.classes = np.array([os.path.basename(os.path.dirname(os.path.dirname(f))) for f in globs])
        self.unique_classes = list(np.unique(self.classes))

        # Gat the list of samples, classes and the sampling rate
        samples = []
        rates = []
        for g in globs:
            print("load train sample: ", g)
            rate, sample = wavfile.read(g, np.dtype)
            sample = mfcc(sample, rate, numcep=mfcc_numcep, nfilt=mfcc_nfilt, nfft=mfcc_nfft)
            samples.append(np.array(sample))
            rates.append(rate)

        #TODO add data augmentation to increase number of training samples

        self.samples = samples
        self.sampling_rates = rates
        self.n_samples = len(self.samples)
    
    # Load the testing data and save all relevant info in arrays
    # TODO Preprocess data before this if neccessary
    def load_test_data(self, mfcc_numcep=24, mfcc_nfilt=26, mfcc_nfft=512): #nfft should be 512 I think but that gives warning messages
        # Get the paths to all the data samples
        globs = glob.glob('../VoxCelebDownload/data/VoxCeleb/vox1_test_wav/*/*/*.wav') #TODO replace id100* with * to load all samples

        # Get the class names from the paths
        self.classes = np.array([os.path.basename(os.path.dirname(os.path.dirname(f))) for f in globs])
        self.unique_classes = list(np.unique(self.classes))

        # Gat the list of samples, classes and the sampling rate
        samples = []
        rates = []
        for g in globs:
            print("load train sample: ", g)
            rate, sample = wavfile.read(g, np.dtype)
            sample = mfcc(sample, rate, numcep=mfcc_numcep, nfilt=mfcc_nfilt, nfft=mfcc_nfft)
            samples.append(np.array(sample))
            rates.append(rate)

        self.samples = samples
        self.sampling_rates = rates
        self.n_samples = len(self.samples)
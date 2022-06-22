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
        self.dsm = 1 #Decrease Sample Number if a fast test for debuging is required

        self.classes = 0
        self.all_classes = 0

        self.samples = 0
        self.sampling_rates = 0
        self.n_samples = 0

    # Returns the sample and class at the given index
    # Can be called as dataset[i] and works with dataloader
    def __getitem__(self, index):
        return torch.from_numpy(self.samples[index]), self.all_classes.index(self.classes[index])

    def __len__(self):
        return self.n_samples
    
    # Load the training data and save all relevant info in arrays
    # TODO Preprocess data before this if neccessary
    def load_train_data(self, mfcc_numcep=24, mfcc_nfilt=26, mfcc_nfft=1103):
        # Get the paths to all the data samples
        globs = glob.glob('data\wavfiles\*\*.wav')

        # Get only every second sample for the training data samples
        # TODO put training and testing data in different folders to make this step easier
        for i, g in enumerate(globs):
            if(i%(2*self.dsm) == 0):
                globs[int(i/(2*self.dsm))] = globs[i]
        globs = globs[:int(len(globs)/(2*self.dsm))]

        # Get the class names from the paths
        self.classes = np.array([os.path.basename(os.path.dirname(f)) for f in globs])
        self.all_classes = list(np.unique(self.classes))

        # Gat the list of samples, classes and the sampling rate
        samples = []
        rates = []
        classes = []
        for i, g in enumerate(globs):
            rate, sample = wavfile.read(g, np.dtype)
            clas = self.classes[i]
            sub_samle_length = int(rate/10)
            print("load train sample: ", g)
            # Split the sample into several 0.1 second long samples to increase number of samples
            # Because the mel frequencies look very different for all 10 instruments cutting down sample length like this should still give good results
            # TODO this takes too long to do every time so do this in preprocessing ahead of time (might not be neccessary with proper dataset)
            for i in range(math.floor(len(sample)/sub_samle_length)-1):
                small_sample = sample[i*sub_samle_length : i*sub_samle_length + sub_samle_length]
                small_sample = mfcc(small_sample, rate, numcep=mfcc_numcep, nfilt=mfcc_nfilt, nfft=mfcc_nfft)
                samples.append(np.array(small_sample))
                rates.append(rate)
                classes.append(clas)

        self.samples = samples
        self.sampling_rates = rates
        self.classes = classes
        self.n_samples = len(self.samples)
    
    # Load the testing data and save all relevant info in arrays
    # TODO Preprocess data before this if neccessary
    def load_test_data(self, mfcc_numcep=24, mfcc_nfilt=26, mfcc_nfft=1103): #nfft should be 512 I think but that gives warning messages
        # Get the paths to all the data samples
        globs = glob.glob('data\wavfiles\*\*.wav')

        # Get only every second sample for the testing data samples
        # TODO put training and testing data in different folders to make this step easier
        for i, g in enumerate(globs):
            if(i%(2*self.dsm) == 1):
                globs[int((i-1)/(2*self.dsm))] = globs[i]
        globs = globs[:int(len(globs)/(2*self.dsm))]

        # Get the class names from the paths
        self.classes = np.array([os.path.basename(os.path.dirname(f)) for f in globs])
        self.all_classes = list(np.unique(self.classes))

        # Gat the list of samples, classes and the sampling rate
        samples = []
        rates = []
        classes = []
        for i, g in enumerate(globs):
            rate, sample = wavfile.read(g, np.dtype)
            clas = self.classes[i]
            sub_samle_length = int(rate/10)
            print("load test sample: ", g)
            # Split the sample into several 0.1 second long samples to increase number of samples
            # Because the mel frequencies look very different for all 10 instruments cutting down sample length like this should still give good results
            # TODO this takes too long to do every time so do this in preprocessing ahead of time (might not be neccessary with proper dataset)
            for i in range(math.floor(len(sample)/sub_samle_length)-1):
                small_sample = sample[i*sub_samle_length : i*sub_samle_length + sub_samle_length]
                small_sample = mfcc(small_sample, rate, numcep=mfcc_numcep, nfilt=mfcc_nfilt, nfft=mfcc_nfft)
                samples.append(np.array(small_sample))
                rates.append(rate)
                classes.append(clas)

        self.samples = samples
        self.sampling_rates = rates
        self.classes = classes
        self.n_samples = len(self.samples)
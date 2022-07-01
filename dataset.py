import glob
import math
import os

import numpy as np
import torch
from python_speech_features import mfcc
from scipy.io import wavfile
from torch.utils.data import Dataset

train_path1 = '../../../../../../../../../data/7hellrie/VoxCeleb/vox1_dev_wav/id1000*/*/00001.wav' #TODO replace id100* with * to load all samples
test_path1 = '../../../../../../../../../data/7hellrie/VoxCeleb/vox1_test_wav/id103*/*/00001.wav'
train_path2 = 'data/VoxCeleb/vox1_dev_wav/wav/id1000*/*/00001.wav'
test_path2 = 'data/VoxCeleb/vox1_test_wav/wav/id103*/*/00001.wav'

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
        globs = glob.glob(train_path1)

        # Get the class names from the paths
        self.classes = np.array([os.path.basename(os.path.dirname(os.path.dirname(f))) for f in globs])
        self.unique_classes = list(np.unique(self.classes))

        # Gat the list of samples, classes and the sampling rate
        samples = []
        rates = []
        classes = []
        for i, g in enumerate(globs):
            print("load train sample: ", g)
            rate, sample = wavfile.read(g, np.dtype)
            clas = self.classes[i]
            sub_sample_length = int(rate * 3)
            # Split the sample into several 3 second long samples to increase number of samples and make samples the same length
            # Because the mel frequencies look very different for all 10 instruments cutting down sample length like this should still give good results
            # TODO maybe do this in preprocessing ahead of time
            for j in range(math.floor(len(sample)/sub_sample_length)):
                small_sample = sample[j*sub_sample_length : j*sub_sample_length + sub_sample_length]
                small_sample = mfcc(small_sample, rate, numcep=mfcc_numcep, nfilt=mfcc_nfilt, nfft=mfcc_nfft)
                samples.append(np.array(small_sample))
                rates.append(rate)
                classes.append(clas)

        #TODO add data augmentation to increase number of training samples

        self.samples = np.array(samples)
        self.sampling_rates = np.array(rates)
        self.classes = np.array(classes)
        self.n_samples = len(self.samples)
    
    # Load the testing data and save all relevant info in arrays
    # TODO Preprocess data before this if neccessary
    def load_test_data(self, mfcc_numcep=24, mfcc_nfilt=26, mfcc_nfft=512): #nfft should be 512 I think but that gives warning messages
        # Get the paths to all the data samples
        globs = glob.glob(test_path1)

        # Get the class names from the paths
        self.classes = np.array([os.path.basename(os.path.dirname(os.path.dirname(f))) for f in globs])
        self.unique_classes = list(np.unique(self.classes))

        # Gat the list of samples, classes and the sampling rate
        samples = []
        rates = []
        classes = []
        for i, g in enumerate(globs):
            print("load test sample: ", g)
            rate, sample = wavfile.read(g, np.dtype)
            clas = self.classes[i]
            sub_sample_length = int(rate * 3)
            # Split the sample into several 3 second long samples to increase number of samples and make samples the same length
            # Because the mel frequencies look very different for all 10 instruments cutting down sample length like this should still give good results
            # TODO maybe do this in preprocessing ahead of time
            for j in range(math.floor(len(sample)/sub_sample_length)):
                small_sample = sample[j*sub_sample_length : j*sub_sample_length + sub_sample_length]
                small_sample = mfcc(small_sample, rate, numcep=mfcc_numcep, nfilt=mfcc_nfilt, nfft=mfcc_nfft)
                samples.append(np.array(small_sample))
                rates.append(rate)
                classes.append(clas)

        self.samples = np.array(samples)
        self.sampling_rates = np.array(rates)
        self.classes = np.array(classes)
        self.n_samples = len(self.samples)
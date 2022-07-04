import glob
import math
import os
import random

import numpy as np
import torch
from python_speech_features import mfcc
from scipy.io import wavfile
from torch.utils.data import Dataset


class Dataset(Dataset): 
    def __init__(self):
        self.unique_labels = []

        self.samples = []
        self.labels = []
        self.n_samples = 0

    # Returns the sample and class at the given index
    # Can be called as dataset[i] and works with dataloader
    def __getitem__(self, index):
        return torch.from_numpy(self.samples[index]), self.unique_labels.index(self.labels[index])

    def __len__(self):
        return self.n_samples
    
    # Load the training data and save all relevant info in arrays
    # TODO Preprocess data before this if neccessary
    def load_train_data(self, mfcc_numcep=24, mfcc_nfilt=26, mfcc_nfft=512, data_folder_path='data', ):
        vox_train_path = data_folder_path + '/VoxCeleb/vox1_dev_wav/id1000*/*/00001.wav' #TODO replace id100* and 00001 with * to load all samples
        self.load_data(mfcc_numcep, mfcc_nfilt, mfcc_nfft, vox_train_path)
    
    # Load the testing data and save all relevant info in arrays
    # TODO Preprocess data before this if neccessary
    def load_test_data(self, mfcc_numcep=24, mfcc_nfilt=26, mfcc_nfft=512, data_folder_path='data'):
        vox_test_path = data_folder_path + '/VoxCeleb/vox1_test_wav/id103*/*/00001.wav' #TODO replace id100* and 00001 with * to load all samples
        self.load_data(mfcc_numcep, mfcc_nfilt, mfcc_nfft, vox_test_path)

    def load_data(self, mfcc_numcep, mfcc_nfilt, mfcc_nfft, data_folder_path):
        # Get the paths to all the data samples
        globs = glob.glob(data_folder_path)

        # Get the class names from the paths
        self.unique_labels = self.unique_labels + [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in globs]
        self.unique_labels = list(np.unique(self.unique_labels))

        # Gat the list of samples, labels and the sampling rate
        for g in globs:
            print("load test sample: ", g)
            rate, sample = wavfile.read(g, np.dtype)

            # Augment the sample with noise and/or reverbaraition
            augmented_samples = self.augment_data(sample, data_folder_path=data_folder_path, num_of_aumented_samples=1)

            clas = os.path.basename(os.path.dirname(os.path.dirname(g)))
            sub_sample_length = int(rate * 3)

            for aug_sample in augmented_samples:
                # Split the sample into several 3 second long samples to increase number of samples and make samples the same length
                # TODO maybe do this in preprocessing ahead of time
                for j in range(math.floor(len(aug_sample)/sub_sample_length)):
                    small_sample = aug_sample[j*sub_sample_length : j*sub_sample_length + sub_sample_length]
                    small_sample = mfcc(small_sample, rate, numcep=mfcc_numcep, nfilt=mfcc_nfilt, nfft=mfcc_nfft)
                    self.samples.append(np.array(small_sample))
                    self.labels.append(clas)
        self.n_samples = len(self.samples)

    def augment_data(self, sample, data_folder_path='data', num_of_aumented_samples=3):
        musan_music_path = data_folder_path + '/musan_split/music/*/*/*.wav'
        musan_speech_path = data_folder_path + '/musan_split/speech/*/*/*.wav'
        rir_mediumroom_path = data_folder_path + '/RIRS_NOISES/simulated_rirs/mediumroom/*/*.wav'
        rir_smallroom_path = data_folder_path + '/RIRS_NOISES/simulated_rirs/smallroom/*/*.wav'
        augmentation_types = [musan_music_path, musan_speech_path, rir_mediumroom_path, rir_smallroom_path]

        augmented_samples = [sample]
        for i in range(num_of_aumented_samples-1):
            augmentation_type = random.choice(augmentation_types)
            augmentation = random.choice(glob.glob(augmentation_type))
            _, aug = wavfile.read(augmentation, np.dtype)
            augmented_samples.append(sample + aug)

        return augmented_samples
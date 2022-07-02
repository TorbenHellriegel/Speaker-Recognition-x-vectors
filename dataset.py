import glob
import math
import os

import numpy as np
import torch
from python_speech_features import mfcc
from scipy.io import wavfile
from torch.utils.data import Dataset

class Dataset(Dataset): 
    def __init__(self):
        self.unique_labels = []
        self.mode = 'x_vector'

        self.samples = []
        self.labels = []
        self.n_samples = 0

        self.x_vectors = []
        self.x_labels = []
        self.n_vectors = 0

    # Returns the sample and class at the given index
    # Can be called as dataset[i] and works with dataloader
    def __getitem__(self, index):
        if(self.mode == 'x_vector'):
            return torch.from_numpy(self.samples[index]), self.unique_labels.index(self.labels[index])
        elif(self.mode == 'plda_classifier'):
            return self.x_vectors[index], self.x_labels[index]

    def __len__(self):
        if(self.mode == 'x_vector'):
            return self.n_samples
        elif(self.mode == 'plda_classifier'):
            return self.n_vectors

    def change_mode(self, new_mode):
        self.mode = new_mode
    
    # Load the training data and save all relevant info in arrays
    # TODO Preprocess data before this if neccessary
    def load_train_data(self, mfcc_numcep=24, mfcc_nfilt=26, mfcc_nfft=512, data_folder_path='data'):
        vox_train_path = data_folder_path + '/VoxCeleb/vox1_dev_wav/id1000*/*/00001.wav' #TODO replace id100* and 00001 with * to load all samples
        musan_music_path = data_folder_path + '/musan_split/music/*/*/*.wav'
        musan_speech_path = data_folder_path + '/musan_split/speech/*/*/*.wav'
        rir_mediumroom_path = data_folder_path + '/RIRS_NOISES/simulated_rirs/mediumroom/*/*.wav'
        rir_smallroom_path = data_folder_path + '/RIRS_NOISES/simulated_rirs/smallroom/*/*.wav'

        # Get the paths to all the data samples
        globs = glob.glob(vox_train_path)

        # Get the class names from the paths
        self.unique_labels = [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in globs]
        self.unique_labels = list(np.unique(self.unique_labels))

        # Gat the list of samples, labels and the sampling rate
        for i, g in enumerate(globs):
            print("load train sample: ", g)
            rate, sample = wavfile.read(g, np.dtype)
            clas = os.path.basename(os.path.dirname(os.path.dirname(g)))
            sub_sample_length = int(rate * 3)
            # Split the sample into several 3 second long samples to increase number of samples and make samples the same length
            # Because the mel frequencies look very different for all 10 instruments cutting down sample length like this should still give good results
            # TODO maybe do this in preprocessing ahead of time
            for j in range(math.floor(len(sample)/sub_sample_length)):
                small_sample = sample[j*sub_sample_length : j*sub_sample_length + sub_sample_length]
                small_sample = mfcc(small_sample, rate, numcep=mfcc_numcep, nfilt=mfcc_nfilt, nfft=mfcc_nfft)
                self.samples.append(np.array(small_sample))
                self.labels.append(clas)

        #TODO add data augmentation to increase number of training samples

        self.n_samples = len(self.samples)
    
    # Load the testing data and save all relevant info in arrays
    # TODO Preprocess data before this if neccessary
    def load_test_data(self, mfcc_numcep=24, mfcc_nfilt=26, mfcc_nfft=512, data_folder_path='data'):
        vox_test_path = data_folder_path + '/VoxCeleb/vox1_test_wav/id103*/*/00001.wav' #TODO replace id100* and 00001 with * to load all samples
        musan_music_path = data_folder_path + '/musan_split/music/*/*/*.wav'
        musan_speech_path = data_folder_path + '/musan_split/speech/*/*/*.wav'
        rir_mediumroom_path = data_folder_path + '/RIRS_NOISES/simulated_rirs/mediumroom/*/*.wav'
        rir_smallroom_path = data_folder_path + '/RIRS_NOISES/simulated_rirs/smallroom/*/*.wav'

        # Get the paths to all the data samples
        globs = glob.glob(vox_test_path)

        # Get the class names from the paths
        self.unique_labels = self.unique_labels + [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in globs]
        self.unique_labels = list(np.unique(self.unique_labels))

        # Gat the list of samples, labels and the sampling rate
        for i, g in enumerate(globs):
            print("load test sample: ", g)
            rate, sample = wavfile.read(g, np.dtype)
            clas = os.path.basename(os.path.dirname(os.path.dirname(g)))
            sub_sample_length = int(rate * 3)
            # Split the sample into several 3 second long samples to increase number of samples and make samples the same length
            # Because the mel frequencies look very different for all 10 instruments cutting down sample length like this should still give good results
            # TODO maybe do this in preprocessing ahead of time
            for j in range(math.floor(len(sample)/sub_sample_length)):
                small_sample = sample[j*sub_sample_length : j*sub_sample_length + sub_sample_length]
                small_sample = mfcc(small_sample, rate, numcep=mfcc_numcep, nfilt=mfcc_nfilt, nfft=mfcc_nfft)
                self.samples.append(np.array(small_sample))
                self.labels.append(clas)

        self.n_samples = len(self.samples)

    def load_train_x_vec(self, x_vec, label):
        self.x_vectors.append(x_vec)
        self.x_labels.append(label)
        self.n_vectors = len(self.x_vectors)

    def load_test_x_vec(self, x_vec, label):
        self.x_vectors.append(x_vec)
        self.x_labels.append(label)
        self.n_vectors = len(self.x_vectors)
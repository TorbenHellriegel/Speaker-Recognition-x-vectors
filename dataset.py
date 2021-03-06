import glob
import os
import random

import numpy as np
import resampy
import torch
from python_speech_features import mfcc
from scipy.io import wavfile
from scipy.signal import fftconvolve
from torch.utils.data import Dataset

EPS = 1e-20

class Dataset(Dataset): 
    def __init__(self, sampling_rate=16000, mfcc_numcep=24, mfcc_nfilt=26, mfcc_nfft=512, data_folder_path='data', augmentations_per_sample=2):
        self.samples = []
        self.labels = []

        self.n_samples = 0
        self.unique_labels = []

        self.data_folder_path = data_folder_path
        self.sampling_rate = sampling_rate
        self.augmentations_per_sample = augmentations_per_sample
        self.mfcc_numcep = mfcc_numcep
        self.mfcc_nfilt = mfcc_nfilt
        self.mfcc_nfft = mfcc_nfft

    # Returns the sample and class at the given index
    # Can be called as dataset[i] and works with dataloader
    def __getitem__(self, index):
        sample_path, augmentation = self.samples[index]
        rate, sample = wavfile.read(sample_path, np.dtype)
        sample = resampy.resample(sample, rate, self.sampling_rate)

        # Augment the sample with noise and/or reverbaraition
        augmented_sample = self.augment_data(sample, augmentation)
        augmented_sample = mfcc(augmented_sample, self.sampling_rate, numcep=self.mfcc_numcep, nfilt=self.mfcc_nfilt, nfft=self.mfcc_nfft)#TODO figure out how mfcc works
        
        return torch.from_numpy(augmented_sample), self.unique_labels.index(self.labels[index])

    def __len__(self):
        return self.n_samples
    
    # Load the training data and save all relevant info in arrays
    def load_train_data(self): #TODO os.path
        vox_train_path = self.data_folder_path + '/VoxCeleb/vox1_dev_wav/*/*/*.wav'
        self.load_data(vox_train_path)
    
    # Load the testing data and save all relevant info in arrays
    def load_test_data(self):
        vox_test_path = self.data_folder_path + '/VoxCeleb/vox1_test_wav//*/*.wav'
        self.load_data(vox_test_path)

    def load_data(self, voxceleb_folder_path):
        # Get the paths to all the data samples
        globs = glob.glob(voxceleb_folder_path)

        # Gat the list of samples and labels
        self.samples = self.samples + [(sample, 'none') for sample in globs]
        self.labels = self.labels + [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in globs]
        for i in range(self.augmentations_per_sample):
            self.samples = self.samples + [(sample, random.choice(['music', 'speech', 'noise', 'rir'])) for sample in globs]
            self.labels = self.labels + [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in globs]

        # Get the num of samples and the unique class names
        self.n_samples = len(self.samples)
        self.unique_labels = list(np.unique(self.labels))

    def augment_data(self, sample, augmentation):
        sample = self.cut_to_sec(sample, 3)

        if(augmentation == 'music'):
            aug_sample = self.augment_musan_music(sample)
        elif(augmentation == 'speech'):
            aug_sample = self.augment_musan_speech(sample)
        elif(augmentation == 'noise'):
            aug_sample = self.augment_musan_noise(sample)
        elif(augmentation == 'rir'):
            aug_sample = self.augment_rir(sample)
        else:
            aug_sample = sample

        aug_sample = aug_sample.astype(np.float64)
        aug_sample -= np.min(aug_sample)
        aug_sample /= np.max(aug_sample)
        return aug_sample

    def cut_to_sec(self, sample, length):
        if(len(sample) < self.sampling_rate*length):
            new_sample = np.pad(sample, (0, self.sampling_rate*length-len(sample)), 'constant', constant_values=(0, 0))
        else:
            start_point = random.randint(0, len(sample) - self.sampling_rate*length)
            new_sample = sample[start_point:start_point + self.sampling_rate*length]
        return new_sample

    def add_with_certain_snr(self, sample, noise, min_snr_db=5, max_snr_db=20):
        sample = sample.astype('int64')
        noise = noise.astype('int64')

        sample_rms = np.sqrt(np.mean(sample**2))
        noise_rms = np.sqrt(np.mean(noise**2))
        wanted_snr = random.randint(min_snr_db, max_snr_db)
        wanted_noise_rms = np.sqrt(sample_rms**2 / 10**(wanted_snr/10))

        new_noise = noise * wanted_noise_rms/(noise_rms+EPS)
        noisy_sample = sample + new_noise
        return noisy_sample

    def augment_musan_music(self, sample):
        musan_music_path = self.data_folder_path + '/musan/music/*/*.wav'
        #print('load sample: augmenting with musan music')

        song_path = random.choice(glob.glob(musan_music_path))
        rate, song = wavfile.read(song_path, np.dtype)
        song = resampy.resample(song, rate, self.sampling_rate)

        song = self.cut_to_sec(song, 3)
        aug_sample = self.add_with_certain_snr(sample, song, min_snr_db=5, max_snr_db=15)
        return aug_sample

    def augment_musan_speech(self, sample):
        musan_speech_path = self.data_folder_path + '/musan/speech/*/*.wav'
        #print('load sample: augmenting with musan speech')

        speaker_path = random.choice(glob.glob(musan_speech_path))
        rate, speakers = wavfile.read(speaker_path, np.dtype)
        speakers = resampy.resample(speakers, rate, self.sampling_rate)
        speakers = self.cut_to_sec(speakers, 3)

        for i in range(random.randint(2, 6)):
            speaker_path = random.choice(glob.glob(musan_speech_path))
            rate, speaker = wavfile.read(speaker_path, np.dtype)
            speaker = resampy.resample(speaker, rate, self.sampling_rate)
            speaker = self.cut_to_sec(speaker, 3)
            speakers = speakers + speaker
            
        aug_sample = self.add_with_certain_snr(sample, speakers, min_snr_db=13, max_snr_db=20)
        return aug_sample

    def augment_musan_noise(self, sample): #TODO leave noise at 1 sec each or overlap?
        musan_noise_path = self.data_folder_path + '/musan/noise/*/*.wav'
        #print('load sample: augmenting with musan noise')
        
        for i in range(3):
            noise_path = random.choice(glob.glob(musan_noise_path))
            rate, noise = wavfile.read(noise_path, np.dtype)
            noise = resampy.resample(noise, rate, self.sampling_rate)
            noise = self.cut_to_sec(noise, 1)
            sample[i:i+self.sampling_rate] = self.add_with_certain_snr(sample[i:i+self.sampling_rate], noise, min_snr_db=0, max_snr_db=15)

        return sample

    def augment_rir(self, sample):
        rir_noise_path = self.data_folder_path + '/RIRS_NOISES/simulated_rirs/*/*/*.wav'
        #print('load sample: augmenting with rir')

        rir_path = random.choice(glob.glob(rir_noise_path))
        _, rir = wavfile.read(rir_path, np.dtype)
        aug_sample = fftconvolve(sample, rir)
        aug_sample = aug_sample / abs(aug_sample).max()

        sample_max = abs(sample).max()
        aug_max = abs(aug_sample).max()
        aug_sample = aug_sample * (sample_max/aug_max)
    
        aug_sample = sample + aug_sample[:len(sample)]
        return aug_sample

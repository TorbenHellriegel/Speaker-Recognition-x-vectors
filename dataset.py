import glob
import math
import os
import random
import resampy
import numpy as np
import torch
from python_speech_features import mfcc
from scipy.io import wavfile
from scipy.signal import convolve
from torch.utils.data import Dataset


class Dataset(Dataset): 
    def __init__(self):
        self.samples = []
        self.labels = []

        self.sampling_rate = 16000
        self.n_samples = 0
        self.unique_labels = []

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
        self.load_data(mfcc_numcep, mfcc_nfilt, mfcc_nfft, data_folder_path, vox_train_path)
    
    # Load the testing data and save all relevant info in arrays
    # TODO Preprocess data before this if neccessary
    def load_test_data(self, mfcc_numcep=24, mfcc_nfilt=26, mfcc_nfft=512, data_folder_path='data'):
        vox_test_path = data_folder_path + '/VoxCeleb/vox1_test_wav/id103*/*/00001.wav' #TODO replace id100* and 00001 with * to load all samples
        self.load_data(mfcc_numcep, mfcc_nfilt, mfcc_nfft, data_folder_path, vox_test_path)

    def load_data(self, mfcc_numcep, mfcc_nfilt, mfcc_nfft, data_folder_path, voxceleb_folder_path):
        # Get the paths to all the data samples
        globs = glob.glob(voxceleb_folder_path)

        # Get the class names from the paths
        self.unique_labels = self.unique_labels + [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in globs]
        self.unique_labels = list(np.unique(self.unique_labels))

        # Gat the list of samples, labels and the sampling rate
        for g in globs:
            print("load sample: ", g)
            rate, sample = wavfile.read(g, np.dtype)
            sample = resampy.resample(sample, rate, self.sampling_rate)
            rate = 16000

            # Augment the sample with noise and/or reverbaraition
            augmented_samples = self.augment_data(sample, data_folder_path=data_folder_path)

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
        augmented_samples = [sample]

        for i in range(num_of_aumented_samples-1):
            augmentation_type = random.randint(0, 2)

            if(augmentation_type == 0):
                aug_sample = self.augment_musan_music(sample, data_folder_path=data_folder_path)
            elif(augmentation_type == 1):
                aug_sample = self.augment_musan_speech(sample, data_folder_path=data_folder_path)
            elif(augmentation_type == 2):
                aug_sample = self.augment_rir(sample, data_folder_path=data_folder_path)
            else:
                aug_sample = sample

            augmented_samples.append(aug_sample)

        return augmented_samples

    def augment_musan_music(self, sample, data_folder_path='data'):
        musan_music_path = data_folder_path + '/musan_old/music/*/*.wav'
        print('load sample: augmenting with musan music')

        song_path = random.choice(glob.glob(musan_music_path))
        rate, song = wavfile.read(song_path, np.dtype)
        song = resampy.resample(song, rate, self.sampling_rate)

        song = self.adjust_augmentation_length(len(sample), song)
        aug_sample = self.add_with_certain_snr(sample, song, min_snr_db=5, max_snr_db=15)
        return aug_sample

    def augment_musan_speech(self, sample, data_folder_path='data'):
        musan_speech_path = data_folder_path + '/musan_old/speech/*/*.wav'
        print('load sample: augmenting with musan speech')

        speakers = np.array([], dtype=np.int16)
        for i in range(random.randint(3, 7)):
            speaker_path = random.choice(glob.glob(musan_speech_path))
            rate, speaker = wavfile.read(speaker_path, np.dtype)
            speaker = resampy.resample(speaker, rate, self.sampling_rate)
            if len(speakers) < len(speaker):
                spkr = speaker.copy()
                spkr[:len(speakers)] += speakers
            else:
                spkr = speakers.copy()
                spkr[:len(speaker)] += speaker
            speakers = spkr
        
        speakers = self.adjust_augmentation_length(len(sample), speakers)
        aug_sample = self.add_with_certain_snr(sample, speakers, min_snr_db=13, max_snr_db=20)
        return aug_sample

    '''def augment_musan_noise(self, sample, data_folder_path='data'):
        musan_noise_path = data_folder_path + '/musan_old/noise/*/*.wav'
        print('load sample: augmenting with musan noise')
        
        for i in range(0, len(sample), self.sampling_rate):
            noise_path = random.choice(glob.glob(musan_noise_path))
            rate, noise = wavfile.read(noise_path, np.dtype)
            noise = resampy.resample(noise, rate, self.sampling_rate)
            noise = self.adjust_augmentation_length(len(sample[i:]), noise)
            sample[i:] = self.add_with_certain_snr(sample[i:], noise, min_snr_db=0, max_snr_db=15)

        return sample'''

    def adjust_augmentation_length(self, sample_length, augmentation):
        if(len(augmentation) > sample_length):
            augmentation = augmentation[:sample_length]
        else:
            new_augmentation = list(augmentation)
            while(sample_length > len(new_augmentation)):
                new_augmentation = new_augmentation + list(augmentation)
            augmentation = np.array(new_augmentation[:sample_length])
        return augmentation

    def add_with_certain_snr(self, sample, noise, min_snr_db=5, max_snr_db=20):
        sample = sample.astype('int64')
        noise = noise.astype('int64')

        sample_rms = np.sqrt(np.mean(sample**2))
        noise_rms = np.sqrt(np.mean(noise**2))
        wanted_snr = random.randint(min_snr_db, max_snr_db)
        wanted_noise_rms = np.sqrt(sample_rms**2 / 10**(wanted_snr/10))

        new_noise = noise * wanted_noise_rms/noise_rms
        noisy_sample = sample + new_noise

        return noisy_sample

    def augment_rir(self, sample, data_folder_path='data'):
        rir_noise_path = data_folder_path + '/RIRS_NOISES_old/simulated_rirs/*/*/*.wav'
        print('load sample: augmenting with rir')

        rir_path = random.choice(glob.glob(rir_noise_path))
        _, rir = wavfile.read(rir_path, np.dtype) #TODO neccessary to adjust the sampling rat for rir?
        
        aug_sample = convolve(sample, rir)
        return aug_sample
import glob
import os
import random

import numpy as np
import resampy
import torch
from python_speech_features import mfcc
from scipy.io import wavfile
from scipy.signal import fftconvolve
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

EPS = 1e-20

class Dataset(Dataset):
    def __init__(self,
                sampling_rate=16000,
                mfcc_numcep=24,
                mfcc_nfilt=26,
                mfcc_nfft=512,
                data_folder_path='data',
                augmentations_per_sample=2):
        self.samples = []
        self.labels = []

        self.n_samples = 0
        self.unique_labels = []

        self.train_samples = []
        self.train_labels = []
        self.val_samples = []
        self.val_labels = []
        self.test_samples = []
        self.test_labels = []

        self.data_folder_path = data_folder_path
        self.sampling_rate = sampling_rate
        self.augmentations_per_sample = augmentations_per_sample
        self.mfcc_numcep = mfcc_numcep
        self.mfcc_nfilt = mfcc_nfilt
        self.mfcc_nfft = mfcc_nfft

    def init_samples_and_labels(self):
        """
        This method initalizes the dataset by collectiong all available train and test data samples.
        The train samples are randomly split into 90% train and 10% validation.
        Even though all samples are collected only the currently active ones will be returned with get.
        To set which samples are currently active call load_data(self, train=False, val=False, test=False)
        and set the wanted samples to true
        """
        vox_train_path = self.data_folder_path + '/VoxCeleb/vox1_dev_wav/*/*/*.wav'
        vox_test_path = self.data_folder_path + '/VoxCeleb/vox1_test_wav/*/*/*.wav'

        # Get the paths to all train and val data samples
        globs = glob.glob(vox_train_path)
        print('collectiong training and validation samples')
        
        # Gat the list of samples and labels
        samples = [(sample, 'none') for sample in globs]
        labels = [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in globs]
        for i in range(self.augmentations_per_sample):
            samples = samples + [(sample, random.choice(['music', 'speech', 'noise', 'rir'])) for sample in globs]
            labels = labels + [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in globs]

        unique_labels = np.unique(labels)
        print('found:')
        print(len(unique_labels), ' unique speakers')
        print(int(len(samples)/(self.augmentations_per_sample+1)), ' voice samples')
        print(len(samples), ' total voice samples including augmentations')
        print('splitting into 90% training and 10% validation')

        skf = StratifiedKFold(n_splits=10, shuffle=True)
        train_index, val_index = [], []
        for traini, vali in skf.split(samples, labels):
            if(len(vali) == int(round(len(samples)/10))):
                train_index = traini
                val_index = vali
        if(len(train_index) <= 1):
            print('StratifiedKFold Failed')
        
        self.train_samples = list(np.array(samples)[train_index])
        self.train_labels = list(np.array(labels)[train_index])
        self.val_samples = list(np.array(samples)[val_index])
        self.val_labels = list(np.array(labels)[val_index])
            
        # Get the paths to all test data samples
        globs = glob.glob(vox_test_path)
        print('collectiong test samples')
        
        # Gat the list of samples and labels
        test_samples = [(sample, 'none') for sample in globs]
        test_labels = [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in globs]
            
        unique_labels = np.unique(test_labels)
        print('found:')
        print(len(unique_labels), ' unique speakers')
        print(len(test_samples), ' voice samples')
        print('DONE collectiong samples')

        self.test_samples = list(np.array(test_samples))
        self.test_labels = list(np.array(test_labels))

    def __getitem__(self, index):
        """
        Returns the MFCC of the sample and class at the given index.

        Parameters
        ----------
        index: int
            The index of the desired sample

        Returns
        -------
        sample: tensor
            The MFCC of the desires sample
        
        label: string
            The label of the sample

        id: string
            The scource directory of the sample (unique for each seperate sample)
        """
        sample_path, augmentation = self.samples[index]
        rate, sample = wavfile.read(sample_path, np.dtype)
        sample = resampy.resample(sample, rate, self.sampling_rate)

        # Augment the sample with noise and/or reverbaraition
        augmented_sample = self.augment_data(sample, augmentation)
        augmented_sample = mfcc(augmented_sample, self.sampling_rate, numcep=self.mfcc_numcep, nfilt=self.mfcc_nfilt, nfft=self.mfcc_nfft)

        label = self.unique_labels.index(self.labels[index])
        id = '/'.join(sample_path.rsplit('/')[-3:])
        
        return torch.from_numpy(augmented_sample), label, id

    def __len__(self):
        """
        Returns the number of active samples.

        Returns
        -------
        n_samples: int
            The number of active samples
        """
        return self.n_samples

    def load_data(self, train=False, val=False, test=False):
        """
        Loads the specified data to be active.

        Parameters
        ----------
        train: bool
            Set if the train samples are supposed to be active
            Default = False
            
        val: bool
            Set if the train samples are supposed to be active
            Default = False
            
        test: bool
            Set if the train samples are supposed to be active
            Default = False
        """
        self.samples = []
        self.labels = []
        self.n_samples = 0
        self.unique_labels = []

        if(train):
            self.samples = self.samples + self.train_samples
            self.labels = self.labels + self.train_labels
        if(val):
            self.samples = self.samples + self.val_samples
            self.labels = self.labels + self.val_labels
        if(test):
            self.samples = self.samples + self.test_samples
            self.labels = self.labels + self.test_labels

        # Get the num of samples and the unique class names
        self.n_samples = len(self.samples)
        self.unique_labels = list(np.unique(self.labels))

    def augment_data(self, sample, augmentation):
        """
        Augment the normalized data sample with a given augmentation.
        If the augmentation is not one of the accepted types returns the unaugmented sample.

        Parameters
        ----------
        sample: ndarray
            The sample that is supposed to be augmented
            
        augmentation: string
            the type of augmentation
            can be: {'music', 'speech', 'noise', 'rir'}

        Returns
        -------
        aug_sample: ndarray
            The normalized augmented sample
        """
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
        """
        Cuts or pads the sample to a certain length.

        Parameters
        ----------
        sample: ndarray
            The sample that is supposed to be cut
            
        length: int
            The lenght in seconds the returned sample should have

        Returns
        -------
        new_sample: ndarray
            The sample with the specefied lenght
        """
        if(len(sample) < self.sampling_rate*length):
            new_sample = np.pad(sample, (0, self.sampling_rate*length-len(sample)), 'constant', constant_values=(0, 0))
        else:
            start_point = random.randint(0, len(sample) - self.sampling_rate*length)
            new_sample = sample[start_point:start_point + self.sampling_rate*length]
        return new_sample

    def add_with_certain_snr(self, sample, noise, min_snr_db=5, max_snr_db=20):
        """
        Adds the noise to the signal with a SNR randomly chosen between min_snr_db and max_snr_db.

        Parameters
        ----------
        sample: ndarray
            The sample that is supposed to be augmented
            
        noise: ndarray
            The noise that is supposed augment the sample
            
        min_snr_db: int
            The minimal SNR in decibil the returned sample should have
            Default = 5
            
        max_snr_db: int
            The maximal SNR in decibil the returned sample should have
            Default = 20

        Returns
        -------
        noisy_sample: ndarray
            The sample with the added noise
        """
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
        """
        Applies background music from the musan dataset to the sample.

        Parameters
        ----------
        sample: ndarray
            The sample that is supposed to be augmented

        Returns
        -------
        aug_sample: ndarray
            The sample with the added noise
        """
        musan_music_path = self.data_folder_path + '/musan/music/*/*.wav'
        #print('load sample: augmenting with musan music')

        song_path = random.choice(glob.glob(musan_music_path))
        rate, song = wavfile.read(song_path, np.dtype)
        song = resampy.resample(song, rate, self.sampling_rate)

        song = self.cut_to_sec(song, 3)
        aug_sample = self.add_with_certain_snr(sample, song, min_snr_db=5, max_snr_db=15)
        return aug_sample

    def augment_musan_speech(self, sample):
        """
        Applies background speech from the musan dataset to the sample.
        3-7 different speakers are added and used as background speech.

        Parameters
        ----------
        sample: ndarray
            The sample that is supposed to be augmented

        Returns
        -------
        aug_sample: ndarray
            The sample with the added noise
        """
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

    def augment_musan_noise(self, sample):
        """
        Applies background noise from the musan dataset to the sample.
        A 1 sec noise clip is added to the sample at 1 sec intervals.

        Parameters
        ----------
        sample: ndarray
            The sample that is supposed to be augmented

        Returns
        -------
        aug_sample: ndarray
            The sample with the added noise
        """
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
        """
        Applies reverbaration from the RIR dataset to the sample.
        The Sample is convolved with a simulated room impulse response.

        Parameters
        ----------
        sample: ndarray
            The sample that is supposed to be augmented

        Returns
        -------
        aug_sample: ndarray
            The sample with the added noise
        """
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

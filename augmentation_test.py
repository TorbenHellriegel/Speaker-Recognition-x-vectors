import glob
import random

import numpy as np
import resampy
from scipy.io import wavfile
from scipy.signal import fftconvolve

EPS = 1e-20

def adjust_augmentation_length(sample_length, augmentation):
    if(len(augmentation) > sample_length):
        augmentation = augmentation[:sample_length]
    else:
        new_augmentation = list(augmentation)
        while(sample_length > len(new_augmentation)):
            new_augmentation = new_augmentation + list(augmentation)
        augmentation = np.array(new_augmentation[:sample_length])

    return augmentation

def add_with_certain_snr(sample, noise, min_snr_db=5, max_snr_db=20):
    sample = sample.astype('int64')
    noise = noise.astype('int64')

    sample_rms = np.sqrt(np.mean(sample**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    wanted_snr = random.randint(min_snr_db, max_snr_db)
    wanted_noise_rms = np.sqrt(sample_rms**2 / 10**(wanted_snr/10))

    new_noise = noise * wanted_noise_rms/(noise_rms+EPS)
    noisy_sample = sample + new_noise
    return noisy_sample

def MUSIC():
    path = random.choice(glob.glob('data/test/music*.wav'))
    rate, song = wavfile.read(path, np.dtype)
    song = resampy.resample(song, rate, 16000)
    song = adjust_augmentation_length(len(sample), song)
    aug_sample = add_with_certain_snr(sample, song, min_snr_db=5, max_snr_db=15)
    aug_sample = aug_sample.astype(np.int16)
    wavfile.write('data/test/voxcelebmusic.wav', 16000, aug_sample)

def SPEECH():
    speakers = np.array([], dtype=np.int16)
    for i in range(3, 7):
        path = random.choice(glob.glob('data/test/speech*.wav'))
        rate, speaker = wavfile.read(path, np.dtype)
        speaker = resampy.resample(speaker, rate, 16000)
        if len(speakers) < len(speaker):
            spkr = speaker.copy()
            spkr[:len(speakers)] += speakers
        else:
            spkr = speakers.copy()
            spkr[:len(speaker)] += speaker
        speakers = spkr
        
    speakers = adjust_augmentation_length(len(sample), speakers)
    aug_sample = add_with_certain_snr(sample, speakers, min_snr_db=5, max_snr_db=10)
    aug_sample = aug_sample.astype(np.int16)
    wavfile.write('data/test/voxcelebspeech.wav', 16000, aug_sample)

def NOISE():
    aug_sample = sample
    for i in range(0, len(sample)-16000, 16000):
        path = random.choice(glob.glob('data/test/noise*.wav'))
        rate, noise = wavfile.read(path, np.dtype)
        noise = resampy.resample(noise, rate, 16000)
        noise = adjust_augmentation_length(16000, noise)
        aug_sample[i:i+16000] = add_with_certain_snr(aug_sample[i:i+16000], noise, min_snr_db=0, max_snr_db=15)
    aug_sample = aug_sample.astype(np.int16)
    wavfile.write('data/test/voxcelebnoise.wav', 16000, aug_sample)
    
def RIR():
    sample_max = abs(sample).max() #save max value to rescale later
    path = random.choice(glob.glob('data/test/Room*.wav'))
    _, rir = wavfile.read(path, np.dtype)

    aug_sample = fftconvolve(sample, rir)
    aug_sample = aug_sample / abs(aug_sample).max() #normalize

    aug_max = abs(aug_sample).max()
    aug_sample = aug_sample * (sample_max/aug_max) #rescale to previous values
    aug_sample = sample + aug_sample[:len(sample)] #add echo to sample

    aug_sample = aug_sample.astype(np.int16)
    wavfile.write('data/test/voxcelebrir.wav', 16000, aug_sample)

#SAMPLE
path = random.choice(glob.glob('data/test/vox0*.wav'))
rate, sample = wavfile.read(path, np.dtype)
sample = resampy.resample(sample, rate, 16000)
rate = 16000

MUSIC()
SPEECH()
NOISE()
RIR()

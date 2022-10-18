import glob
import random

import numpy as np
import resampy
from scipy.io import wavfile
from scipy.signal import fftconvolve

EPS = 1e-20

def cut_to_sec(sample, length):
    if(len(sample) < 16000*length):
        new_sample = np.pad(sample, (0, 16000*length-len(sample)), 'constant', constant_values=(0, 0))
    else:
        start_point = random.randint(0, len(sample) - 16000*length)
        new_sample = sample[start_point:start_point + 16000*length]
    return new_sample

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

def MUSIC(sample):
    path = random.choice(glob.glob('data/test_augmentations/music*.wav'))
    rate, song = wavfile.read(path, np.dtype)
    song = resampy.resample(song, rate, 16000)
    song = cut_to_sec(song, 3)
    aug_sample = add_with_certain_snr(sample, song, min_snr_db=5, max_snr_db=15)
    aug_sample = aug_sample.astype(np.int16)
    wavfile.write('data/test_augmentations/voxcelebmusic.wav', 16000, aug_sample)

def SPEECH(sample):
    speaker_path = random.choice(glob.glob('data/test_augmentations/speech*.wav'))
    rate, speakers = wavfile.read(speaker_path, np.dtype)
    speakers = resampy.resample(speakers, rate, 16000)
    speakers = cut_to_sec(speakers, 3)
    for i in range(random.randint(2, 6)):
        speaker_path = random.choice(glob.glob('data/test_augmentations/speech*.wav'))
        rate, speaker = wavfile.read(speaker_path, np.dtype)
        speaker = resampy.resample(speaker, rate, 16000)
        speaker = cut_to_sec(speaker, 3)
        speakers = speakers + speaker
        
    aug_sample = add_with_certain_snr(sample, speakers, min_snr_db=5, max_snr_db=10)
    aug_sample = aug_sample.astype(np.int16)
    wavfile.write('data/test_augmentations/voxcelebspeech.wav', 16000, aug_sample)

def NOISE(sample):
    aug_sample = sample
    for i in range(3):
        noise_path = random.choice(glob.glob('data/test_augmentations/noise*.wav'))
        rate, noise = wavfile.read(noise_path, np.dtype)
        noise = resampy.resample(noise, rate, 16000)
        noise = cut_to_sec(noise, 1)
        sample[i*16000:(i+1)*16000] = add_with_certain_snr(sample[i*16000:(i+1)*16000], noise, min_snr_db=0, max_snr_db=15)
    aug_sample = aug_sample.astype(np.int16)
    wavfile.write('data/test_augmentations/voxcelebnoise.wav', 16000, aug_sample)
    
def RIR(sample):
    path = random.choice(glob.glob('data/test_augmentations/Room*.wav'))
    _, rir = wavfile.read(path, np.dtype)

    aug_sample = fftconvolve(sample, rir)
    aug_sample = aug_sample / abs(aug_sample).max() #normalize

    sample_max = abs(sample).max() #save max value to rescale later
    aug_max = abs(aug_sample).max()
    aug_sample = aug_sample * (sample_max/aug_max) #rescale to previous values
    
    aug_sample = sample + aug_sample[:len(sample)] #add echo to sample
    aug_sample = aug_sample.astype(np.int16)
    wavfile.write('data/test_augmentations/voxcelebrir.wav', 16000, aug_sample)

#SAMPLE
path = random.choice(glob.glob('data/test_augmentations/voxceleb*.wav'))
rate, sample = wavfile.read(path, np.dtype)
sample = resampy.resample(sample, rate, 16000)
sample = cut_to_sec(sample, 3)
rate = 16000
wavfile.write('data/test_augmentations/voxceleb3sec.wav', 16000, sample)

MUSIC(sample)
SPEECH(sample)
RIR(sample)
NOISE(sample)

print('done')
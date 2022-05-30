#! .\bchlr-venv\scripts\python.exe

from asyncio.windows_events import NULL
import os
import glob
import numpy as np
from statistics import mean
from sklearn.utils import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
from python_speech_features import mfcc

def build_rand_feat():
    X = []
    y = []
    global_min = float('inf')
    global_max = -float('inf')
    for _ in range(n_samples):
        rand_class = np.random.choice(classes, p=prob_dist)
        file = np.random.choice([sample for i, sample in enumerate(df[:, 0]) if df[i, 1] == rand_class])
        rate, wav = wavfile.read('data/clean/' + file)
        rand_index = np.random.randint(0, wav.shape[0] - config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft).T
        global_min = min(np.amin(X_sample), global_min)
        global_max = max(np.amax(X_sample), global_max)
        X.append(X_sample if config.mode == 'conv' else X_sample.T) #make sure the data has the format that the nn model needs
        y.append(classes.index(rand_class))
    X, y = np.array(X), np.array(y) #no sure if this is such a good idea
    X = (X - global_min) / (global_max - global_min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y_hot = torch.from_numpy(y).type(torch.int64)
    y_hot = F.one_hot(y_hot, num_classes=10)
    return X, y, y_hot

class conv_model(nn.Module):   
    def __init__(self):
        super(conv_model, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    
def get_recurrent_model():
    return 1

# TODO ?
class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate = 16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)

# Create array with [[filename], [instrument], [length]]
glob = glob.glob('data\wavfiles\*\*.wav')
df = np.transpose([[os.path.basename(f) for f in glob], #get all files in all wavefile directorys
        [os.path.basename(os.path.dirname(f)) for f in glob], #get the names of all the corresponding directories
        [0 for f in glob]]) #add in an array of extra 0 to later fill with length values

# Calculate the length in [[filename], [instrument], [length]]
# Also calculate sampling rate
sr = 0
for i in range(len(df)):
    rate, signal = wavfile.read('data/clean/' + df[i, 0]) #read the wav file
    df[i, 2] = signal.shape[0]/rate #calculate the length of each sample and save it in the array
    sr = rate

# Calculate the distrubution for each instrument (mean leangth)
classes = list(np.unique(df[:, 1])) #get a list of all unique classes
class_dist = np.zeros_like(classes)
for i, c in enumerate(classes):
    class_list = np.where(df[:, 1] == c) #get all positions in the array with the same class c
    class_length = [l for l in df[class_list, 2]] #get the length of all samples with the same class
    class_length = [float(l) for l in class_length[0]] #char to float
    class_dist[i] = mean(class_length) #calculate the mean to get the class distribution

# TODO what are we doing here?
n_samples = 2 * int(df[:, 2].astype(float).sum()/0.1  /  100) #TODO dont decrease sample number
prob_dist = class_dist.astype(float) / class_dist.astype(float).sum()
#print(np.random.choice(classes, p=prob_dist))

# Plot the class distribution data
fig, ax = plt.subplots()
ax.set_title('Class Distribution')
ax.pie(class_dist, labels=classes, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
#plt.show()

# Set which kind of nn to use. For practice it's 'conv' or 'time' for actual use it's 'tdnn'
config = Config(mode='conv')

# TODO ?
if config.mode == 'conv':
    X, y_flat, y_hot = build_rand_feat()
    input_shape = (X.shape[1], X.shape[2], 1)
    model = conv_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.07)
    criterion = nn.CrossEntropyLoss()

elif config.mode == 'time':
    X, y_flat, y_hot = build_rand_feat()
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()

# Calculate a weight value for each class. The lower the number of samples from that class the higer the value
#class_weight = compute_class_weight('balanced', classes=np.unique(y_flat), y=y_flat)
#model.fit(X, y_hot, epochs=10, batch_size=32, shuffle=True, class_weight=class_weight)

print(model)

def train(epoch):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = torch.autograd.Variable(torch.from_numpy(X[0])), torch.autograd.Variable(torch.from_numpy(y_flat))

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    
    # prediction for training and validation set
    output_train = model(x_train)

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    train_losses.append(loss_train)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch%2 == 0:
        # printing the validation loss
        print('Epoch : ',epoch+1, '\t', 'loss :', loss_train)

# defining the number of epochs
n_epochs = 6
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
# training the model
for epoch in range(n_epochs):
    train(epoch)
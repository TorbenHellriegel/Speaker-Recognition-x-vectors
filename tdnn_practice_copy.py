#! .\bchlr-venv\scripts\python.exe

from glob import glob
import os
import glob
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from scipy.io import wavfile
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from python_speech_features import mfcc

# Push calculation to gpu if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define variables
input_size = 13
hidden_size = 500
num_classes = 10
num_epochs = 1
batch_size = 25
learning_rate = 0.001
PATH = './trained models/tdnntest.pth'

# TODO adjust based on what the dataset looks like
class Dataset(Dataset): 
    def __init__(self):
        self.filenames = 0
        self.classes = 0
        self.all_classes = 0

        self.samples = 0
        self.sampling_rates = 0
        self.n_samples = 0

    # Returns the sample at the given index with the predefined transform
    # Can be called as dataset[i]
    def __getitem__(self, index):
        return torch.from_numpy(self.samples[index]), self.all_classes.index(self.classes[index])

    def __len__(self):
        return self.n_samples
    
    # Load the training data and save all relevant info in arrays
    # TODO PREPROCESS DATA BEFORE THIS
    def load_train_data(self):
        # Get the paths to all the data samples
        globs = glob.glob('data\wavfiles\*\*.wav')

        # Get only the testing data samples
        # TODO put training and testing data in different folders to make this step easier
        for i, g in enumerate(globs):
            if(i%2 == 0):
                globs[int(i/2)] = globs[i]
        globs = globs[:int(len(globs)/2)]

        # Get the filenames and class names from the paths
        self.filenames = np.array([os.path.basename(f) for f in globs])
        self.classes = np.array([os.path.basename(os.path.dirname(f)) for f in globs])
        self.all_classes = list(np.unique(self.classes))

        # Gat the list of samples and the sampling rate
        samples = []
        rates = []
        for g in globs:
            rate, sample = wavfile.read(g, np.dtype)
            rand_index = np.random.randint(0, sample.shape[0] - 1600)
            sample = sample[rand_index:rand_index+1600]
            sample = mfcc(sample, rate, numcep=13, nfilt=26, nfft=512)
            samples.append(np.array(sample))
            rates.append(rate)

        self.samples = samples
        self.sampling_rates = rates
        self.n_samples = len(self.samples)
    
    # Load the testing data and save all relevant info in arrays
    # TODO PREPROCESS DATA BEFORE THIS
    def load_test_data(self):
        # Get the paths to all the data samples
        globs = glob.glob('data\wavfiles\*\*.wav')

        # Get only the testing data samples
        # TODO put training and testing data in different folders to make this step easier
        for i, g in enumerate(globs):
            if(i%2 == 1):
                globs[int((i-1)/2)] = globs[i]
        globs = globs[:int(len(globs)/2)]

        # Get the filenames and class names from the paths
        self.filenames = np.array([os.path.basename(f) for f in globs])
        self.classes = np.array([os.path.basename(os.path.dirname(f)) for f in globs])
        self.all_classes = list(np.unique(self.classes))

        # Gat the list of samples and the sampling rate
        samples = []
        rates = []
        for g in globs:
            rate, sample = wavfile.read(g, np.dtype)
            rand_index = np.random.randint(0, sample.shape[0] - 1600)
            sample = sample[rand_index:rand_index+1600]
            sample = mfcc(sample, rate, numcep=13, nfilt=26, nfft=512)
            samples.append(np.array(sample))
            rates.append(rate)

        self.samples = samples
        self.sampling_rates = rates
        self.n_samples = len(self.samples)

# Make two datasets for training and testing with different samples
train_dataset = Dataset()
train_dataset.load_train_data()
test_dataset = Dataset()
test_dataset.load_test_data()

# Set up dataloader for easy access to shuffled data batches
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define Neural Network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)
# Alternatively load an existing model dictionary
#model.load_state_dict(torch.load(PATH))
#model.eval()

# Setup loss and optimizer
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(train_data_loader)
model = model.float()

for epoch in range(num_epochs):
    for i, (images, labels) in (enumerate(train_data_loader)):
        # Reshape image data
        images = images
        labels_hot = F.one_hot(labels, num_classes=10)

        # Forward pass
        outputs = model(images.float())
        loss = criteria(outputs, labels_hot)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')

# Save the model dictionary
#torch.save(model.state_dict(), PATH)

# Testing
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0

#     for images, labels in test_data_loader:
#         # Reshape image data
#         images = images
#         labels = labels

#         outputs = model(images)

#         _, predictions = torch.max(outputs.data, 1)
#         n_samples += labels.size(0)
#         n_correct += (predictions == labels).sum().item()

#     accuracy = 100.0 * n_correct / n_samples
#     print(f'accuracy = {accuracy}')
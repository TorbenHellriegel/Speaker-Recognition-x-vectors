#! .\bchlr-venv\scripts\python.exe

import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
from python_speech_features import mfcc

# Push calculation to gpu if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set important variables
batch_size = 100
input_size = 24
hidden_size = 512
num_classes = 10
learning_rate = 0.001
num_epochs = 5

# LOAD DATA

# TODO adjust based on what the dataset looks like
class Dataset(Dataset): 
    def __init__(self):
        self.dsm = 1 #Decrease Sample Number if a fast test for debuging is required

        self.classes = 0
        self.all_classes = 0

        self.samples = 0
        self.sampling_rates = 0
        self.n_samples = 0

    # Returns the sample and class at the given index
    # Can be called as dataset[i] and works with dataloader
    def __getitem__(self, index):
        return torch.from_numpy(self.samples[index]), self.all_classes.index(self.classes[index])

    def __len__(self):
        return self.n_samples
    
    # Load the training data and save all relevant info in arrays
    # TODO Preprocess data before this if neccessary
    def load_train_data(self):
        # Get the paths to all the data samples
        globs = glob.glob('data\wavfiles\*\*.wav')

        # Get only every second sample for the training data samples
        # TODO put training and testing data in different folders to make this step easier
        for i, g in enumerate(globs):
            if(i%(2*self.dsm) == 0):
                globs[int(i/(2*self.dsm))] = globs[i]
        globs = globs[:int(len(globs)/(2*self.dsm))]

        # Get the class names from the paths
        self.classes = np.array([os.path.basename(os.path.dirname(f)) for f in globs])
        self.all_classes = list(np.unique(self.classes))

        # Gat the list of samples, classes and the sampling rate
        samples = []
        rates = []
        classes = []
        for i, g in enumerate(globs):
            rate, sample = wavfile.read(g, np.dtype)
            clas = self.classes[i]
            sub_samle_length = int(rate/10)
            print("load train sample: ", g)
            # Split the sample into several 0.1 second long samples to increase number of samples
            # Because the mel frequencies look very different for all 10 instruments cutting down sample length like this should still give good results
            # TODO this takes too long to do every time so do this in preprocessing ahead of time (might not be neccessary with proper dataset)
            for i in range(math.floor(len(sample)/sub_samle_length)-1):
                small_sample = sample[i*sub_samle_length : i*sub_samle_length + sub_samle_length]
                small_sample = mfcc(small_sample, rate, numcep=input_size, nfilt=26, nfft=1103)
                samples.append(np.array(small_sample))
                rates.append(rate)
                classes.append(clas)

        self.samples = samples
        self.sampling_rates = rates
        self.classes = classes
        self.n_samples = len(self.samples)
    
    # Load the testing data and save all relevant info in arrays
    # TODO Preprocess data before this if neccessary
    def load_test_data(self):
        # Get the paths to all the data samples
        globs = glob.glob('data\wavfiles\*\*.wav')

        # Get only every second sample for the testing data samples
        # TODO put training and testing data in different folders to make this step easier
        for i, g in enumerate(globs):
            if(i%(2*self.dsm) == 1):
                globs[int((i-1)/(2*self.dsm))] = globs[i]
        globs = globs[:int(len(globs)/(2*self.dsm))]

        # Get the class names from the paths
        self.classes = np.array([os.path.basename(os.path.dirname(f)) for f in globs])
        self.all_classes = list(np.unique(self.classes))

        # Gat the list of samples, classes and the sampling rate
        samples = []
        rates = []
        classes = []
        for i, g in enumerate(globs):
            rate, sample = wavfile.read(g, np.dtype)
            clas = self.classes[i]
            sub_samle_length = int(rate/10)
            print("load test sample: ", g)
            # Split the sample into several 0.1 second long samples to increase number of samples
            # Because the mel frequencies look very different for all 10 instruments cutting down sample length like this should still give good results
            # TODO this takes too long to do every time so do this in preprocessing ahead of time (might not be neccessary with proper dataset)
            for i in range(math.floor(len(sample)/sub_samle_length)-1):
                small_sample = sample[i*sub_samle_length : i*sub_samle_length + sub_samle_length]
                small_sample = mfcc(small_sample, rate, numcep=input_size, nfilt=26, nfft=1103) #nfft should be 512 I think but that gives warning messages
                samples.append(np.array(small_sample))
                rates.append(rate)
                classes.append(clas)

        self.samples = samples
        self.sampling_rates = rates
        self.classes = classes
        self.n_samples = len(self.samples)

# Make two datasets for training and testing with different samples
train_dataset = Dataset()
train_dataset.load_train_data()
test_dataset = Dataset()
test_dataset.load_test_data()

# Set up dataloader for easy access to shuffled data batches
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# TODO Maybe visualize select data samples for images for the thesis

# DEFINE NEURAL NETWORK

class TDNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TDNN, self).__init__()

        self.time_context_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size), #TODO add time context somehow
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1500),
            nn.LeakyReLU()
        )
        
        self.segment_layer6 = nn.Sequential(
            nn.Linear(3000, hidden_size),
            nn.LeakyReLU(),
        )
        
        self.segment_layer7 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )

        self.output = nn.Linear(hidden_size, num_classes)

        self.softmax = nn.Softmax() #TODO use softmax? nn.CrossEntropyLoss() appearently already includes a softmax
        
    def forward(self, x):
        out = self.time_context_layers(x)
        # Satistic pooling layer
        mean = torch.mean(out, 1)
        stand_dev = torch.std(out, 1)
        out = torch.cat((mean, stand_dev), 1)
        out = self.segment_layer6(out)
        out = self.segment_layer7(out)
        out = self.output(out)  #TODO use softmax? nn.CrossEntropyLoss() appearently already includes a softmax
        return out #TODO also return the output of the segment_layers for the x-vector

model = TDNN(input_size, hidden_size, num_classes)
# Alternatively load an existing model dictionary
#model.load_state_dict(torch.load(PATH))
#model.eval()
model = model.to(device)
model = model.float()

# Set up loss and optimizer
criterion = nn.CrossEntropyLoss() #also applies softmax
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_steps = len(train_data_loader)

for epoch in range(num_epochs):
    model.train()
    for i, (samples, labels) in enumerate(train_data_loader):
        # Adjust data
        samples.requires_grad = True
        samples = samples.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(samples.float())
        _, predictions = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {total_steps}, loss = {loss.item():.4f}')

# Save the model dictionary
#torch.save(model.state_dict(), PATH)

# Testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for sapmles, labels in test_data_loader:

        outputs = model(samples.float())

        _, predictions = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predictions == labels).sum().item()

    accuracy = 100.0 * n_correct / n_samples
    print(f'accuracy = {accuracy}')
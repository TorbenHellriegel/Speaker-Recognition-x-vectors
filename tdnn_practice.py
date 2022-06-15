#! .\bchlr-venv\scripts\python.exe

import os
import glob
import tqdm
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
frames = 1 #TODO ?
num_classes = 10
learning_rate = 0.001
num_epochs = 1

# LOAD DATA

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
        classes = []
        for i, g in enumerate(globs):
            rate, sample = wavfile.read(g, np.dtype)
            clas = self.classes[i]
            print("load train: ", g)
            for i in range(math.floor(len(sample)/1600)-1):
                small_sample = sample[i*1600:i*1600+1600]
                small_sample = mfcc(small_sample, rate, numcep=input_size, nfilt=26, nfft=1103)
                samples.append(np.array(small_sample))
                rates.append(rate)
                classes.append(clas)

        self.samples = samples
        self.sampling_rates = rates
        self.classes = classes
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
        classes = []
        for i, g in enumerate(globs):
            rate, sample = wavfile.read(g, np.dtype)
            clas = self.classes[i]
            print("load test: ", g)
            for i in range(math.floor(len(sample)/1600)-1):
                small_sample = sample[i*1600:i*1600+1600]
                small_sample = mfcc(small_sample, rate, numcep=input_size, nfilt=26, nfft=1103)
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
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Maybe visualize select data samples



# DEFINE NEURAL NETWORK

class TDNN(nn.Module):
    def __init__(self, input_size, hidden_size, frames, num_classes):
        super(TDNN, self).__init__()

        self.time_context_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size), #TODO add time context somehow
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1500),
            nn.ReLU()
        )
        
        self.segment_layers = nn.Sequential(
            nn.Linear(1500, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.output = nn.Linear(hidden_size, num_classes)

        self.softmax = nn.Softmax() #TODO how to use softmax? #TODO use softmax? not if i use nn.CrossEntropyLoss()
        
    def forward(self, x):
        out = self.time_context_layers(x)
        ### Stat Pool
        # mean = torch.mean(out,1)
        # std = torch.std(out,1)
        # out = torch.cat((mean,std),1)
        out = self.segment_layers(out)
        out = self.output(out) #TODO use softmax? not if i use nn.CrossEntropyLoss()
        return out #TODO maybe also return seg6 and seg7 for the x-vector?

model = TDNN(input_size, hidden_size, frames, num_classes).to(device)
model = model.float()
# Alternatively load an existing model dictionary
#model.load_state_dict(torch.load(PATH))
#model.eval()

# Set up loss and optimizer
criterion = nn.CrossEntropyLoss() #also applies softmax
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(train_data_loader)

for epoch in range(num_epochs):
    model.train()
    for i, (samples, labels) in enumerate(train_data_loader):
        # Reshape data
        samples.requires_grad = True
        labels_hot = F.one_hot(labels, num_classes=10)

        # Forward pass
        outputs = model(samples.float())
        loss = criterion(outputs, labels_hot)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')

# Save the model dictionary
#torch.save(model.state_dict(), PATH)

# Testing
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0

#     for sapmles, labels in test_data_loader:
#         # Reshape  data
#         labels_hot = F.one_hot(labels, num_classes=10)

#         outputs = model(samples.float())

#         _, predictions = torch.max(outputs.data, 1)
#         n_samples += labels_hot.size(0)
#         n_correct += (predictions == labels_hot).sum().item()

#     accuracy = 100.0 * n_correct / n_samples
#     print(f'accuracy = {accuracy}')
#! .\bchlr-venv\scripts\python.exe

import os
import glob
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
batch_size = 1
input_size = 24
hidden_size = 512
frames = 1 #TODO ?
num_classes = 10
learning_rate = 0.01
num_epochs = 2

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
        for g in globs:
            rate, sample = wavfile.read(g, np.dtype)
            rand_index = np.random.randint(0, sample.shape[0] - 1600)
            sample = sample[rand_index:rand_index+1600]
            sample = mfcc(sample, rate, numcep=input_size, nfilt=26, nfft=512)
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
            sample = mfcc(sample, rate, numcep=input_size, nfilt=26, nfft=512)
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

# Maybe visualize select data samples



# DEFINE NEURAL NETWORK

class TDNNbroken(nn.Module):
    def __init__(self, input_size, hidden_size, frames, num_classes):
        super(TDNNbroken, self).__init__()

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

        self.pooling_layer = nn.Sequential(
            nn.MaxPool2d(1500 * frames, 3000),
            #nn.ReLU()
        )
        
        self.segment_layers = nn.Sequential(
            nn.Linear(3000, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.ReLU()
        )

        self.softmax = nn.Sequential( #TODO use softmax? not if i use nn.CrossEntropyLoss()
            nn.Softmax(), #TODO how to use softmax?
            #nn.ReLU()
        )
        
    def forward(self, x):
        x = self.time_context_layers(x)
        x = nn.Linear(1500, num_classes) #TODO change later
        #x = self.pooling_layer(x)
        #x = self.segment_layers(x)
        #x = self.softmax(x) #TODO use softmax? not if i use nn.CrossEntropyLoss()
        return x #TODO maybe also return seg6 and seg7 for the x-vector?

class TDNN(nn.Module):
    def __init__(self, input_size, hidden_size, frames, num_classes):
        super(TDNN, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

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
    for i, (samples, labels) in enumerate(train_data_loader):
        # Reshape data
        samples = samples
        labels_hot = F.one_hot(labels, num_classes=10)

        # Forward pass
        outputs = model(samples.float())
        loss = criterion(outputs, labels_hot)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')

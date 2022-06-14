#! .\bchlr-venv\scripts\python.exe

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# TensorBoard add-on for mnist
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist")

#Just stuff----------------------------------------------------------------------
'''
x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y * y * 2
print(z)

v = torch.tensor([1, 1, 1])
z.backward(v)
print(x.grad)'''

#Bachpropagation----------------------------------------------------------------------
'''
x=torch.tensor(1.0)
y=torch.tensor(2)

w=torch.tensor(1.0, requires_grad=True)
print(w)

y_hat = w*x
loss = y_hat-y
loss = loss**2

loss.backward()
w_grad = w.grad
print(w_grad/3)

for i in range(10):
    w = torch.tensor(w-w_grad/3, requires_grad=True)
    print(w)

    y_hat = w*x
    loss = y_hat-y
    loss = loss**2

    loss.backward()
    w_grad = w.grad
    print(w_grad/3)'''

#Gradient calculation numpy vs torch----------------------------------------------------------------------
'''
# Training Pipeline Torch:
# 0. Prepare data
# 1. Design Model (input size, output size, forward pass layers)
# 2. Costruct loss and optimizer
# 3. Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights


# function f = 2*x
# x = np.array([1, 2, 3, 4]) <<<NUMPY>>>
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
# y = np.array([2, 4, 6, 8]) <<<NUMPY>>>
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float)

# w = 0.0 <<<NUMPY>>>
# w = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
n_samples, n_features = x.shape
input_size = output_zize = n_features

# model = nn.Linear(input_size, output_zize)
# the model is usually more complicated
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.lin(x)
model = LinearRegression(input_size, output_zize)

# model prediction <<<NUMPY>>>
# def forward(x):
#     return w*x

# loss (MSE = 1/N * (w*x - y)**2) <<<NUMPY>>>
# def loss(y, y_predic):
#     return ((y_predic-y)**2).mean()

# gradient MSE' = dJ/dw = 1/N * ((2x^2 * w) - (2xy)) = 1/N * 2x * (xw - y) <<<NUMPY>>>
# def gradient(x, y, y_predic):
#     return np.dot(2*x, y_predic-y).mean()



# training
learning_rate = 0.01
n_iter = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

x_test = torch.tensor([5], dtype=torch.float)
print(f'Prediction before training: f(5) = {model(x_test).item():.3f}   Correct answer = 10')

for i in range(n_iter):
    # prediction = forward pass
    # y_predic = forward(x)
    y_predic = model(x)
    
    # loss
    l = loss(y, y_predic)
    
    # gradient = backward pass
    # grad_dw = gradient(x, y, y_predic) <<<NUMPY>>>
    l.backward()

    # update weights
    # w -= learning_rate * grad_dw <<<NUMPY>>>
    # with torch.no_grad():
    #     w -= learning_rate * w.grad
    optimizer.step()
    
    # empty gradients
    # w.grad.zero_()
    optimizer.zero_grad()

    if i % 10 == 0:
        [w, b] = model.parameters()
        print(f'i {i+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(x_test).item():.3f}   Correct answer = 10')'''

#Practical example linear regression----------------------------------------------------------------------
'''
# 0. Prepare data
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = x.shape


# 1. Design Model (input size, output size, forward pass layers)
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2. Costruct loss and optimizer
learning_rate = 0.01
criteria = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3. Training loop
n_iter = 100
for i in range(n_iter):
    #forward pass and loss
    y_predic = model(x)
    loss = criteria(y_predic, y)

    # backwardpass
    loss.backward()

    # update
    optimizer.step()

    optimizer.zero_grad()

    if (i+1) % 10 == 0:
        print(f'iteration: {i+1}, loss = {loss.item():.4f}')

# plot
predicted = model(x).detach().numpy()
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, predicted, 'b')
plt.show()'''

#Practical example logistic regression----------------------------------------------------------------------
'''
# 0. Prepare data
bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target

n_samples, n_features = x.shape

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# scale
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1. Design Model f = wx + b, sigmoid at the end
input_size = n_features
output_size = 1

class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predic = torch.sigmoid(self.linear(x))
        return y_predic

model = LogisticRegression(n_features)


# 2. Costruct loss and optimizer
learning_rate = 0.01
criteria = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 3. Training loop
n_iter = 1000
for i in range(n_iter):
    #forward pass and loss
    y_predic = model(x_train)
    loss = criteria(y_predic, y_train)

    # backwardpass
    loss.backward()

    # update
    optimizer.step()

    optimizer.zero_grad()

    if (i+1) % 100 == 0:
        print(f'iteration: {i+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predic = model(x_test)
    y_predic_class = y_predic.round()
    accuracy = y_predic_class.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {accuracy:.4f}')'''

#Practical example Dataset DataLoader----------------------------------------------------------------------
'''
class WineDataset(Dataset):

    def __init__(self):
        xy = np.loadtxt('./data/wine.txt', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

batch_size = 4
dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

num_epochs = 2
total_samples = len(dataset)
n_iteration = math.ceil(total_samples/batch_size)
print(total_samples, n_iteration)

for epoch in range(num_epochs):
    for i, (inputs, lables) in enumerate(dataloader):
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iteration}, inputs {inputs.shape}')'''

#Practical example Dataset Transforms----------------------------------------------------------------------
'''
class WineDataset(Dataset):

    def __init__(self, transform=None):
        xy = np.loadtxt('./wine.txt', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        self.x = xy[:, 1:]
        self.y = xy[:, [0]]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

dataset = WineDataset(transform=ToTensor())

first_data = dataset[0]
features, lables = first_data
print(features)
print(type(features), type(lables))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)

first_data = dataset[0]
features, lables = first_data
print(features)
print(type(features), type(lables))'''

#Practical example MNIST classification----------------------------------------------------------------------

# Push calculation to gpu if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define variables
input_size = 28**2
hidden_size = 500
num_classes = 10
num_epochs = 1
batch_size = 100
learning_rate = 0.001
PATH = './trained models/mnist model.pth'

# Load MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

example = iter(train_data_loader)
example_data, _ = example.next()

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i][0], cmap='gray')
#plt.show()

# Add images to TensorBoard
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('mnist_images', img_grid)

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

# Add neural network graph to TensorBoard
writer.add_graph(model, example_data.reshape(-1, 28*28))

# Extra variables to add information about training data to TensorBoard
running_loss = 0.0
running_correct = 0

# Training loop
n_total_steps = len(train_data_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in tqdm(enumerate(train_data_loader)):
        # Reshape image data
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criteria(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulating information on training data for TensorBoard
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')

            # Add loss and accuracy information to TensorBoard
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('accuracy', running_correct / 100, epoch * n_total_steps + i)
            running_loss = 0.0
            running_correct = 0

# Save the model dictionary
#torch.save(model.state_dict(), PATH)

# Extra variables to add precision recall curve TensorBoard
labels_tb = []
preds_tb = []

# Testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for images, labels in test_data_loader:
        # Reshape image data
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predictions == labels).sum().item()

        # Accumulating information for precision recall curve for TensorBoard
        labels_tb.append(predictions)
        class_prediction = [F.softmax(output, dim=0) for output in outputs]
        preds_tb.append(class_prediction)
    
    # Add precision recall curve to TensorBoard
    labels_tb = torch.cat(labels_tb)
    preds_tb = torch.cat([torch.stack(batch) for batch in preds_tb])
    classes = range(10)
    for i in classes:
        labels_i = labels_tb == i
        preds_i = preds_tb[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
    writer.close()
    #sys.exit()

    accuracy = 100.0 * n_correct / n_samples
    print(f'accuracy = {accuracy}')

#Practical example CIFAR10 classification----------------------------------------------------------------------
'''
# Push calculation to gpu if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-Parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

# Transform images from [0, 1] to normalized tesors [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR10
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define Convolutional Neural Network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet().to(device)

# Setup loss and optimizer
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(train_data_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_data_loader):
        # Reshape image data
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criteria(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

# Save the trained Neural Network
# print('Finished Training')
# PATH = './cnn.pth'
# torch.save(model.state_dict(), PATH)

# testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_data_loader:
        # Reshape image data
        images = images.to(device)
        labels = labels.to(device)

        # Use trained model on testin images
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predictions == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predictions[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    
    # Determine accuracy
    accuracy = 100.0 * n_correct / n_samples
    print(f'accuracy = {accuracy} %')

    for i in range(10):
        accuracy = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'accuracy of {classes[i]}: {accuracy} %')'''


# Training Pipeline Torch:
# 0. Prepare data
# 1. Design Model (input size, output size, forward pass layers)
# 2. Costruct loss and optimizer
# 3. Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

'''
# Push calculation to gpu if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define variables TODO for testing make everything less
input_size = 120
hidden_size = 512
frames = 10 #TODO get number of frames somehow
num_classes = 10 #TODO number of different speakers needs to be calculated
num_epochs = 1
batch_size = 100
learning_rate = 0.001
PATH = './trained models/x vector model.pth'

# Load data
class WineDataset(Dataset): #TODO adjust based on what the dataset looks like
    def __init__(self, transform=None):
        xy = np.loadtxt('./tododataset.txt', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        self.x = xy[:, 1:]
        self.y = xy[:, [0]]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

dataset = WineDataset(transform=ToTensor())

train_dataset = dataset #TODO wir brauchen nur die hälfte oder so
test_dataset = dataset #TODO wir brauchen nur die hälfte oder so

train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#TODO visualize data somehow
example = iter(train_data_loader)
example_data, example_labels = example.next()
# Add images to TensorBoard
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('x_vector_images', img_grid)

# Define Neural Network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, frames, num_classes):
        super(NeuralNet, self).__init__()
        self.f1 = nn.Linear(input_size, hidden_size) #TODO add layer context somehow
        self.f2 = nn.Linear(hidden_size*3, hidden_size)
        self.f3 = nn.Linear(hidden_size*3, hidden_size)
        self.f4 = nn.Linear(hidden_size, hidden_size)
        self.f5 = nn.Linear(hidden_size, 1500)
        self.pool = nn.MaxPool2d(1500 * frames, 3000)
        self.seg6 = nn.Linear(3000, hidden_size)
        self.seg7 = nn.Linear(hidden_size, hidden_size)
        #TODO softmax? no if i use nn.CrossEntropyLoss()
    
    def forward(self, x):
        out = F.relu(self.f1(x))
        out = F.relu(self.f2(out))
        out = F.relu(self.f3(out))
        out = F.relu(self.f4(out))
        out = F.relu(self.f5(out))
        out = self.pool(out)
        out = F.relu(self.seg6(out))
        out = F.relu(self.seg7(out))
        #TODO softmax? no if i use nn.CrossEntropyLoss()
        return out #TODO maybe also return seg6 and seg7 for the x-vector

model = NeuralNet(input_size, hidden_size, frames, num_classes).to(device)
# Alternatively load an existing model dictionary
#model.load_state_dict(torch.load(PATH))
#model.eval()

# Setup loss and optimizer
criterion = nn.CrossEntropyLoss() #also applies softmax
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Add neural network graph to TensorBoard
writer.add_graph(model, example_data.reshape(-1, 28*28))

# Extra variables to add information about training data to TensorBoard
running_loss = 0.0
running_correct = 0

# Training loop
n_total_steps = len(train_data_loader)

for epoch in range(num_epochs):
    for i, (sapmles, labels) in enumerate(train_data_loader):
        # Reshape sample data
        sapmles = sapmles.reshape(-1, 120).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(sapmles)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulating information on training data for TensorBoard
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')

            # Add loss and accuracy information to TensorBoard
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('accuracy', running_correct / 100, epoch * n_total_steps + i)
            running_loss = 0.0
            running_correct = 0

# Save the model dictionary
#torch.save(model.state_dict(), PATH)

# Extra variables to add precision recall curve TensorBoard
labels_tb = []
preds_tb = []

# Testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for sapmles, labels in test_data_loader:
        # Reshape image data
        sapmles = sapmles.reshape(-1, 120).to(device)
        labels = labels.to(device)

        outputs = model(sapmles)

        _, predictions = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predictions == labels).sum().item()

        # Accumulating information for precision recall curve for TensorBoard
        labels_tb.append(predictions)
        class_prediction = [F.softmax(output, dim=0) for output in outputs]
        preds_tb.append(class_prediction)
    
    # Add precision recall curve to TensorBoard
    labels_tb = torch.cat(labels_tb)
    preds_tb = torch.cat([torch.stack(batch) for batch in preds_tb])
    classes = range(10)
    for i in classes:
        labels_i = labels_tb == i
        preds_i = preds_tb[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
    writer.close()
    #sys.exit()

    accuracy = 100.0 * n_correct / n_samples
    print(f'accuracy = {accuracy}')'''
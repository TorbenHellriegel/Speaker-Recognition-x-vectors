import torch
import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Dataset
from neural_net import NeuralNet


class Config:
    def __init__(self, batch_size=100, input_size=24, hidden_size=512, num_classes=10, learning_rate=0.001,
                num_epochs=5, path='./trained_models/x_vector_model.pth'):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.path = path

def load_data():
    # Make two datasets for training and testing with different samples
    train_dataset = Dataset()
    train_dataset.load_train_data()
    test_dataset = Dataset()
    test_dataset.load_test_data()

    # Set up dataloader for easy access to shuffled data batches
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    return train_data_loader, test_data_loader

def setup_model(load_existing=False):
    model = NeuralNet(config.input_size, config.hidden_size, config.num_classes)
    # Maybe load an existing pretrained model dictionary
    if(load_existing):
        model.load_state_dict(torch.load(config.path))
        model.eval()
    model = model.to(device)
    model = model.float()

    # Set up loss and optimizer
    criterion = nn.CrossEntropyLoss() #also applies softmax
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    return model, criterion, optimizer

def train():
    total_steps = len(train_data)
    for epoch in range(config.num_epochs):
        model.train()
        for i, (samples, labels) in enumerate(train_data):
            # Adjust data
            samples.requires_grad = True
            samples = samples.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(samples.float())
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 1 == 0:
                print(f'epoch {epoch+1} / {config.num_epochs}, step {i+1} / {total_steps}, loss = {loss.item():.4f}')
            #TODO return extra data for the graphs in the thesis

def test():
    with torch.no_grad():
        n_correct = 0
        n_samples = 0

        for samples, labels in test_data:
            samples = samples.to(device)
            labels = labels.to(device)

            outputs = model(samples.float())

            _, predictions = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predictions == labels).sum().item()

        accuracy = 100.0 * n_correct / n_samples
        print(f'accuracy = {accuracy}')
        #TODO return extra data for the graphs in the thesis

if __name__ == "__main__":
    # Push calculation to gpu if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(torch.cuda.is_available())

    # Set important variables
    config = Config()

    # Load data
    train_data, test_data = load_data()

    # TODO Maybe visualize select data samples for images for the thesis

    # Define neural network
    model, criterion, optimizer = setup_model(load_existing=True)

    # Training loop
    train()

    # Save the model dictionary
    torch.save(model.state_dict(), config.path)

    # Testing
    test()
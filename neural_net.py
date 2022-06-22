import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

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
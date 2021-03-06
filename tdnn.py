import torch
import torch.nn as nn


class TdnnLayer(nn.Module):
    def __init__(self, input_size=24, output_size=512, context=[0]):
        super(TdnnLayer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.context = context

        self.linear = nn.Linear(input_size*len(context), output_size)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(output_size)

    def forward(self, x):

        x_context = get_time_context(x, self.context)
        x = torch.cat(x_context, 2)
        x = self.linear(x)
        x = self.relu(x)

        x = x.transpose(1,2)
        x = self.norm(x)
        x = x.transpose(1,2)

        return x

def get_time_context(x, c=[0]):
    l = len(c) - 1
    xc =   [x[:, c[l]+cc:c[0]+cc, :]
            if cc!=c[l] else
            x[:, c[l]+cc:, :]
            for cc in c]
    return xc

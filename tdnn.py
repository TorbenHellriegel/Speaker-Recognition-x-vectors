import torch
import torch.nn as nn


class TdnnLayer(nn.Module):
    def __init__(self, input_size=24, output_size=512, context=[0], batch_norm=True, dropout_p=0.0):
        """
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Structure inspired by https://github.com/cvqluu/TDNN/blob/master/tdnn.py
        """
        super(TdnnLayer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.context = context
        self.batch_norm = batch_norm
        self.dropout_p = dropout_p

        self.linear = nn.Linear(input_size*len(context), output_size)
        self.relu = nn.ReLU()
        if(self.batch_norm):
            self.norm = nn.BatchNorm1d(output_size)
        if(self.dropout_p):
            self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):

        x_context = get_time_context(x, self.context)
        x = torch.cat(x_context, 2)
        x = self.linear(x)
        x = self.relu(x)
        
        if(self.dropout_p):
            x = self.drop(x)

        if(self.batch_norm):
            x = x.transpose(1,2)
            x = self.norm(x)
            x = x.transpose(1,2)

        return x

def get_time_context(x, c=[0]):
    """
    Returns x with the applied time context. For this the surrounding time frames are concatenated together.
    For example an input of shape (100, 10) with context [-1,0,1] would become (98,30).
    Visual example:
    x=          c=          result=
    [[1,2],                 
    [3,4],                  [[1,2,3,4,5,6],
    [5,6],      [-1,0,1]    [3,4,5,6,7,8],
    [7,8],                  [5,6,7,8,9,0]]
    [9,0]]                  
    """
    l = len(c) - 1
    xc =   [x[:, c[l]+cc:c[0]+cc, :]
            if cc!=c[l] else
            x[:, c[l]+cc:, :]
            for cc in c]
    return xc

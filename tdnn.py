import torch
import torch.nn as nn
import torch.nn.functional as F

class TdnnLayer(nn.Module):
    
    def __init__(self, input_dim=24,  output_dim=512, context_size=5, stride=1, dilation=1):
        '''
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TdnnLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.context_size = context_size
        self.stride = stride
        self.dilation = dilation
      
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_dim)
        
    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        print("-----------------------------------")
        print("context: ", self.context_size, ", ", self.dilation)
        print("input x shape: ", x.shape)

        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        print("x unsqueezed shape: ", x.shape)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
                        x, 
                        (self.context_size, self.input_dim), 
                        stride=(1,self.input_dim), 
                        dilation=(self.dilation,1)
                    )

        print("x unfolded shape: ", x.shape)

        x = x.transpose(1,2)
        print("x transposed shape: ", x.shape)
        x = self.kernel(x)
        x = self.nonlinearity(x)

        x = x.transpose(1,2)
        x = self.bn(x)
        x = x.transpose(1,2)

        return x
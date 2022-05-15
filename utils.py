import numpy as np
import torch.nn as nn
import torch


def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = np.dot(A_inv, u)
    A_inv -= np.outer(Au, Au)/(1+np.dot(u.T, Au))
    return A_inv

def relu_s(x, s, c):
    return c*(torch.relu(x)**s)


class Model(nn.Module):
    """Template for fully connected neural network for scalar approximation.
    """
    def __init__(self,
                 input_size=1,
                 hidden_size=2,
                 n_layers=1,
                 p=0.0,
                 seed=42,
                 s=1
                 ):
        super(Model, self).__init__()

        self.n_layers = n_layers
        c_sigma = 2/np.prod([(2*k + 1) for k in range(s)])
        self.scale_cnst = np.sqrt(c_sigma) #/hidden_size

        if self.n_layers == 1:
            self.layers = [nn.Linear(input_size, 1)]
        else:
            size = [input_size] + [hidden_size, ] * (self.n_layers-1) + [1]
            self.layers = [nn.Linear(size[i], size[i+1], bias=False) for i in range(self.n_layers)]
            torch.manual_seed(seed)
            for l in self.layers:
                nn.init.normal_(l.weight, mean=0.0, std=np.sqrt(1/hidden_size))
        self.layers = nn.ModuleList(self.layers)

        # dropout layer
        self.dropout = nn.Dropout(p=p)

        # define the activation function
        self.activation = relu_s #nn.ReLU()
        self.s = s

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.dropout(self.activation(self.layers[i](x), self.s, self.scale_cnst))
        x = self.layers[-1](x)
        return x

    

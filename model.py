import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(10, 16)
        self.linear2 = nn.Linear(16, 1)

    def forward(self, inputs):
        x = F.tanh(self.linear1(inputs))
        x = self.linear2(x)
        return x

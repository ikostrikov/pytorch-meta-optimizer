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

# A helper class that keeps track of meta updates
# It's done by replacing parameters with variables and applying updates to
# them.


class MetaModel(Model):

    def reset(self):
        for module in self.children():
            module._parameters['weight'] = Variable(
                module._parameters['weight'].data)
            module._parameters['bias'] = Variable(
                module._parameters['bias'].data)

    def copy_params_from(self, model):
        for modelA, modelB in zip(self.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)

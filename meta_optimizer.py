from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class MetaOptimizer(nn.Module):

    def __init__(self, hidden_size):
        super(MetaOptimizer, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(1, hidden_size)

        self.lstm = nn.LSTMCell(hidden_size, hidden_size)

        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear2.weight.data.mul_(0.1)
        self.linear2.bias.data.fill_(0.0)

        self.reset_lstm()

    def reset_lstm(self):
        self.hx = Variable(torch.zeros(1, self.hidden_size))
        self.cx = Variable(torch.zeros(1, self.hidden_size))

    def forward(self, inputs):
        initial_size = inputs.size()
        x = inputs.view(-1, 1)
        x = F.tanh(self.linear1(x))

        if x.size(0) != self.hx.size(0):
            self.hx = self.hx.expand(x.size(0), self.hx.size(1))
            self.cx = self.hx.expand(x.size(0), self.cx.size(1))

        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        x = self.hx

        x = self.linear2(x)
        x = x.view(*initial_size)
        return x

    def meta_update(self, meta_model, model_with_grads):
        # First we need to create a flat version of parameters and gradients
        weight_shapes = []
        bias_shapes = []

        params = []
        grads = []

        for module in meta_model.children():
            weight_shapes.append(list(module._parameters['weight'].size()))
            bias_shapes.append(list(module._parameters['bias'].size()))

            params.append(module._parameters['weight'].view(-1))
            params.append(module._parameters['bias'].view(-1))

        for module in model_with_grads.children():
            grads.append(module._parameters['weight'].grad.view(-1))
            grads.append(module._parameters['bias'].grad.view(-1))

        flat_params = torch.cat(params)
        flat_grads = torch.cat(grads)

        # Meta update itself
        flat_params = flat_params + self(flat_grads)

        # Restore original shapes
        offset = 0
        for i, module in enumerate(meta_model.children()):
            weight_flat_size = reduce(mul, weight_shapes[i], 1)
            bias_flat_size = reduce(mul, bias_shapes[i], 1)

            module._parameters['weight'] = flat_params[
                offset:offset + weight_flat_size].view(*weight_shapes[i])
            module._parameters['bias'] = flat_params[
                offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(*bias_shapes[i])

            offset += weight_flat_size + bias_flat_size

        # Finally, copy values from the meta model to the normal one.
        meta_model.copy_params_to(model_with_grads)

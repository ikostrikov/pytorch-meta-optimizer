import argparse
import operator
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import get_batch
from meta_optimizer import MetaModel, MetaOptimizer, FastMetaOptimizer
from model import Model
from torch.autograd import Variable
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--optimizer_steps', type=int, default=100, metavar='N',
                    help='number of meta optimizer steps (default: 100)')
parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='step at which it truncates bptt (default: 20)')
parser.add_argument('--updates_per_epoch', type=int, default=10, metavar='N',
                    help='updates per epoch (default: 100)')
parser.add_argument('--max_epoch', type=int, default=10000, metavar='N',
                    help='number of epoch (default: 10000)')
parser.add_argument('--hidden_size', type=int, default=10, metavar='N',
                    help='hidden size of the meta optimizer (default: 10)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

assert args.optimizer_steps % args.truncated_bptt_step == 0

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

def main():
    # Create a meta optimizer that wraps a model into a meta model
    # to keep track of the meta updates.
    meta_model = Model()
    if args.cuda:
        meta_model.cuda()

    meta_optimizer = FastMetaOptimizer(MetaModel(meta_model), args.num_layers, args.hidden_size)
    if args.cuda:
        meta_optimizer.cuda()

    optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-3)

    for epoch in range(args.max_epoch):
        decrease_in_loss = 0.0
        final_loss = 0.0
        train_iter = iter(train_loader)
        for i in range(args.updates_per_epoch):

            # Sample a new model
            model = Model()
            if args.cuda:
                model.cuda()

            x, y = next(train_iter)
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            # Compute initial loss of the model
            f_x = model(x)
            initial_loss = F.nll_loss(f_x, y)

            for k in range(args.optimizer_steps // args.truncated_bptt_step):
                # Keep states for truncated BPTT
                meta_optimizer.reset_lstm(
                    keep_states=k > 0, model=model, use_cuda=args.cuda)

                loss_sum = 0
                prev_loss = torch.zeros(1)
                if args.cuda:
                    prev_loss = prev_loss.cuda()
                for j in range(args.truncated_bptt_step):
                    x, y = next(train_iter)
                    if args.cuda:
                        x, y = x.cuda(), y.cuda()
                    x, y = Variable(x), Variable(y)

                    # First we need to compute the gradients of the model
                    f_x = model(x)
                    loss = F.nll_loss(f_x, y)
                    model.zero_grad()
                    loss.backward()

                    # Perfom a meta update using gradients from model
                    # and return the current meta model saved in the optimizer
                    meta_model = meta_optimizer.meta_update(model, loss.data)

                    # Compute a loss for a step the meta optimizer
                    f_x = meta_model(x)
                    loss = F.nll_loss(f_x, y)

                    loss_sum += (loss - Variable(prev_loss))

                    prev_loss = loss.data

                # Update the parameters of the meta optimizer
                meta_optimizer.zero_grad()
                loss_sum.backward()
                for param in meta_optimizer.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            # Compute relative decrease in the loss function w.r.t initial
            # value
            decrease_in_loss += loss.data[0] / initial_loss.data[0]
            final_loss += loss.data[0]

        print("Epoch: {}, final loss {}, average final/initial loss ratio: {}".format(epoch, final_loss / args.updates_per_epoch,
                                                                       decrease_in_loss / args.updates_per_epoch))

if __name__ == "__main__":
    main()

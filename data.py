import torch


def get_batch(batch_size):
    x = torch.randn(batch_size, 10)
    x = x - 2 * x.pow(2)
    y = x.sum(1)
    return x, y

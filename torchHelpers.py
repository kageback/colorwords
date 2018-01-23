import torch
from torch.autograd import Variable


def long_var(tensor):
    return Variable(torch.LongTensor(tensor)).cuda()


def float_var(tensor):
        return Variable(torch.FloatTensor(tensor)).cuda()
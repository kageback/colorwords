import torch
from torch.autograd import Variable


def long_var(tensor, cuda):
    var = Variable(torch.LongTensor(tensor))
    if cuda:
        var = var.cuda()
    return var


def float_var(tensor, cuda):
    var = Variable(torch.FloatTensor(tensor))
    if cuda:
        var = var.cuda()
    return var
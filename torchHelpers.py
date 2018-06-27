import torch
from torch.autograd import Variable


def long_var(tensor, use_cuda=None):
    return cuda(Variable(torch.LongTensor(tensor)), use_cuda)


def float_var(tensor, use_cuda=None):
    return cuda(Variable(torch.FloatTensor(tensor)), use_cuda)


def cuda(var, use_cuda=None):
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()

    if use_cuda:
        return var.cuda()
    else:
        return var.cpu()

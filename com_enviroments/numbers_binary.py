import os
import numpy as np
import requests
import re
import torch
import time
from ast import literal_eval
#import matplotlib.pyplot as plt
from com_enviroments.BaseEnviroment import BaseEnviroment
from sklearn.linear_model import LinearRegression

class NumberEnvironment(BaseEnviroment):
    def __init__(self, data_dim=100, smoothing=False) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.smoothing = smoothing
        self.max_length = len(np.binary_repr(self.data_dim))
        self.samp = 'freq'
        self.numbers = np.array(list(range(self.data_dim))) + 1
        self.num_use_dist = self.get_use_dist(data_dim, smoothing)

    def full_batch(self):
        full_batch = self.binary_rep(self.numbers)
        return torch.LongTensor(self.numbers), torch.FloatTensor(full_batch)

    def mini_batch(self, batch_size=10):
        if self.samp == 'freq':
            # Sample from categorical
            batch = np.random.choice(a=self.numbers, size=batch_size, replace=True, p=self.num_use_dist)
            binary_batch = self.binary_rep(batch)
        else:
            batch = self.binary_rep(np.random.randint(0, self.data_dim, batch_size))
        return batch, binary_batch

    def binary_rep(self, batch):
        binary_batch = np.zeros([batch.shape[0], self.max_length])
        for i in range(batch.shape[0]):
            binary_batch[i, :] = np.binary_repr(batch[i], width=self.max_length)
        return binary_batch

    def bin2int(b):
        return b.dot(2 ** np.arange(b.size)[::-1])

    def abs_dist(self, guess, target):

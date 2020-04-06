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
        self.samp = 'freq'
        self.numbers = np.array(list(range(self.data_dim)))
        self.num_use_dist = self.get_use_dist(data_dim, smoothing)

    def full_batch(self):
        return torch.LongTensor(self.numbers), torch.FloatTensor(np.expand_dims(self.numbers, axis=1)) + 1

    def mini_batch(self, batch_size=10):
        if self.samp == 'freq':
            # Sample from categorical
            batch = np.expand_dims(np.random.choice(a=self.numbers, size=batch_size, replace=True, p=self.num_use_dist), axis=1)
        else:
            batch = np.expand_dims(np.random.randint(0, self.data_dim, batch_size), axis=1)
        return batch, batch + 1

    def sim_index(self, num_a, num_b):
        return self.data_dim - np.sqrt(np.power(num_a-num_b, 2))

    def get_use_dist(self, data_dim, smoothing):
        fname = 'data/ngram_2.npy'
        if os.path.isfile(fname):
            data = np.load(fname)
        else:
            download_ngram(100)
            data = np.load(fname)
        data = data[0:data_dim]
        if smoothing:
            data = np.log(data)
            reg = LinearRegression().fit(np.asarray(list(range(1, data_dim + 1))).reshape(-1, 1), data)
            smoothed = reg.predict(np.asarray(list(range(1, data_dim + 1))).reshape(-1, 1))
            data = np.exp(smoothed)
        data = data / np.sum(data)
        return data


import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

def get_freq(N):
    word_list = []
    for i in range(1, N+1):
        synset = wn.synsets(str(i))[0]
        word_list.append(synset.name().split('.')[0].replace('_', ' '))
    return(word_list)


def download_ngram(N):
    fname = 'data/ngram_2.npy'
    if os.path.isfile(fname):
        data = np.load(fname)
    else:
        data = []
        word_list = get_freq(N)
        for query in range(1, 101):
        # for i in range(int(len(word_list)/10)):
        #     numbers=[word_list[i*10 : (i+1)*10]]
        #     query = ','.join([str(n) for n in numbers])
            params = dict(content=query, year_start=1999, year_end=2000,
                          corpus=15, smoothing=3)
            req = requests.get('http://books.google.com/ngrams/graph', params=params)
            res = re.findall('var data = (.*?);\\n', req.text)
            print(req)
            while res == []:
                time.sleep(360)
                req = requests.get('http://books.google.com/ngrams/graph', params=params)
                print(req)
                res = re.findall('var data = (.*?);\\n', req.text)
            data += [qry['timeseries'][1] for qry in literal_eval(res[0])]
            print(len([qry['timeseries'][1] for qry in literal_eval(res[0])]))
            time.sleep(2)
        data = np.array(data)
        print(data)
        np.save(fname, data)

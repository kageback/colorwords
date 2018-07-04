import os
import numpy as np
import com_enviroments.getngrams as ngram
import requests
import re
from ast import literal_eval
import matplotlib.pyplot as plt
from com_enviroments.BaseEnviroment import BaseEnviroment

class NumberEnvironment(BaseEnviroment):
    def __init__(self) -> None:
        super().__init__()
        self.data_dim = 100
        self.samp = 'freq'
        self.numbers = np.array(list(range(self.data_dim)))
        self.num_use_dist = self.get_use_dist()
        plt.plot(range(100), self.num_use_dist)
        plt.savefig('fig/wrd_dist.png')
    def full_batch(self):
        return self.numbers, np.expand_dims(self.numbers, axis=1)

    def mini_batch(self, batch_size=10):
        if self.samp == 'freq':
            # Sample from categorical
            batch = np.expand_dims(np.random.choice(a=self.numbers, size=batch_size, replace=True, p=self.num_use_dist), axis=1)
        else:
            batch = np.expand_dims(np.random.randint(0, self.data_dim, batch_size), axis=1)
        return batch, batch

    def sim_np(self, num_a, num_b):
        return self.data_dim - np.sqrt(np.power(num_a-num_b, 2))

    def get_use_dist(self):
        fname = 'data/num_use_dist.npy'
        if os.path.isfile(fname):
            data = np.load(fname)
        else:
            data = []
            for i in range(10):
                numbers = range(i*10,(i+1)*10)
                query = ','.join([str(n) for n in numbers])
                params = dict(content=query, year_start=1999, year_end=2000,
                              corpus=15, smoothing=3)
                req = requests.get('http://books.google.com/ngrams/graph', params=params)
                res = re.findall('var data = (.*?);\\n', req.text)
                data += [qry['timeseries'][1] for qry in literal_eval(res[0])]
            data = np.array(data)
            data /= data.sum()
            np.save(fname, data)
        return data

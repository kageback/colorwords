import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import os
import requests
import re
from ast import literal_eval
import time
import numpy as np
def get_freq(N):
    word_list = []
    for i in range(1, N+1):
        synset = wn.synsets(str(i))[0]
        word_list.append(synset.name().split('.')[0].replace('_', ' '))
    return(word_list)
get_freq(10)


def get_ngram_dist(N):
    fname = 'data/ngram_2.npy'
    if os.path.isfile(fname):
        data = np.load(fname)
    else:
        data = []
        word_list = get_freq(N)
        for query in range(1,101):
            
        # for i in range(int(len(word_list)/10)):
        #     numbers=[word_list[i*10 : (i+1)*10]]
        #     query = ','.join([str(n) for n in numbers])
            params = dict(content=query, year_start=1999, year_end=2000,
                          corpus=15, smoothing=1)
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

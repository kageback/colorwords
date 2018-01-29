import pandas as pd
import numpy as np


wcs_path = 'data/'
color_chips = pd.read_csv(wcs_path + 'cnum-vhcm-lab-new.txt', sep='\t')
cielab_map = color_chips[['L*', 'a*', 'b*']].values


def all_colors():
    return color_chips.index.values, cielab_map


def batch(batch_size = 10):
    batch = color_chips.sample(n=batch_size, replace=True)

    return batch.index.values, batch[['L*', 'a*', 'b*']].values


def color_dim():
    return len(color_chips)


def chip_index2CIELAB(color_codes):
    return cielab_map[color_codes]


# Evaluation metrics

def communication_cost_regier(V, sum_over_whole_s=False, norm_over_s=False, weight_by_size=False):

    s = {}
    for i in V.keys():
        s[i] = 0
        for j in V.keys():
            if V[i]['word'] == V[j]['word']:
                s[i] += sim(i, j)


    l = {}
    for t in V.keys():
        z = 0
        cat_size = 0
        for i in V.keys():

            if sum_over_whole_s or V[i]['word'] == V[t]['word']:
                z += s[i]
                cat_size += 1
        l[t] = s[t]/z
        if weight_by_size:
            l[t] *= cat_size/len(V)

    if norm_over_s:
        l_z=0
        for x in l.values():
            l_z += x
        for i in l.keys():
            l[i] /= l_z

    # debug code to check it l sums to one
    l_z=0
    for x in l.values():
        l_z += x

    E = 0
    for t in V.keys():
        E += -np.log2(l[t])
    E = E / len(V)

    return E


def wellformedness(V):
    Sw = 0
    for i in V.keys():
        for j in V.keys():
            if V[i]['word'] == V[j]['word']:
                Sw += sim(i, j)
    Da = 0
    for i in V.keys():
        for j in V.keys():
            if V[i]['word'] != V[j]['word']:
                Da += 1- sim(i, j)
    W = Sw + Da
    return W


def combined_criterion(V):
    def inc_dict(dict, key, increment):
        if key in dict.keys():
            dict[key] += increment
        else:
            dict[key] = increment


    cat_sizes = {}
    for v in V.values():
        inc_dict(cat_sizes,v['word'],1)

    n = len(cat_sizes)

    cost = 0
    for w in cat_sizes.keys():
        cost += cat_sizes[w]*np.log2(cat_sizes[w])/n

    ## CCP part
    CCP = 0
    for i in V.keys():
        for j in V.keys():
            if V[i]['word'] != V[j]['word']:
                CCP += 2*sim(i, j)-1

    return cost + CCP

def min_k_cut_cost(V, k):
    def xrange(start, stop):
        return range(start, stop + 1)

    C = {}
    for i in xrange(1, k):
        C[i] = []

    for chip_index in V.keys():
        C[V[chip_index]['word']+1].append(chip_index)

    cost = 0
    for i in xrange(1, k-1):
        for j in xrange(i+1, k):
            for v1 in C[i]:
                for v2 in C[j]:
                    cost += sim(v1, v2)
    return cost


def sim(chip_index_x, chip_index_y, c=0.001):

    color_x = cielab_map[chip_index_x]
    color_y = cielab_map[chip_index_y]

    # CIELAB distance 76 (euclidean distance)
    d = np.linalg.norm(color_x - color_y, 2)

    # Regier color similarity
    return np.exp(-c * np.power(d, 2))


# Printing

def print_cnum(t):
    return str(t['#cnum'].values[0])


def print_color_map(f=print_cnum, pad=3):
    # print x axsis
    print(''.ljust(pad), end="")
    for x in range(41):
        print(str(x).ljust(pad), end="")
    print('')

    # print color codes
    for y in list('ABCDEFGHIJ'):
        print(y.ljust(pad), end="")
        for x in range(41):
            t = color_chips.loc[(color_chips['H'] == x) & (color_chips['V'] == y)]
            if len(t) == 0:
                s = ''
            elif len(t) == 1:
                s = f(t)
            else:
                raise TabError()

            print(s.ljust(pad), end="")
        print('')







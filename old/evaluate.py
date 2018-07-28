import numpy as np

import torchHelpers as th


import itertools


def combined_criterion(V, sim):
    n, cat_sizes = compute_term_usage(V)

    cost = 0
    for w in cat_sizes.keys():
        cost += cat_sizes[w]*np.log2(cat_sizes[w])/n

    ## CCP part
    CCP = 0
    for i in V.keys():
        for j in V.keys():
            if V[i] != V[j]:
                CCP += 2*sim(i, j)-1

    return cost + CCP


def min_k_cut_cost(V, k, sim):
    def xrange(start, stop):
        return range(start, stop + 1)

    C = {}
    for i in xrange(1, k):
        C[i] = []

    for chip_index in V.keys():
        C[V[chip_index]+1].append(chip_index)

    cost = 0
    for i in xrange(1, k-1):
        for j in xrange(i+1, k):
            for v1 in C[i]:
                for v2 in C[j]:
                    cost += sim(v1, v2)
    return cost

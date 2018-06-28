import numpy as np
import torch.nn.functional as F
import torch.nn.functional as F

import torchHelpers as th
from com_enviroments import wcs

def color_graph_V(a, env):
    V = {}
    a = th.cuda(a)
    chip_indices, colors = env.all_colors()
    colors = th.float_var(colors)

    probs = a(perception=colors)
    _, words = probs.max(1)

    for chip_index in chip_indices:
        V[chip_index] = words[chip_index].cpu().data[0]

    return V

def compute_gibson_cost(a, wcs):
    chip_indices, colors = wcs.all_colors()
    colors = th.float_var(colors, False)
    color_terms = th.long_var(range(a.msg_dim), False)

    p_WC = a(perception=colors).t().data.numpy()
    p_CW = F.softmax(a(msg=color_terms), dim=1).data.numpy()

    S = -np.diag(np.matmul(p_WC.transpose(), (np.log2(p_CW))))

    avg_S = S.sum() / len(S)  # expectation assuming uniform prior


    # debug code
    # s = 0
    # c = 43
    # for w in range(a.msg_dim):
    #     s += -p_WC[w, c]*np.log2(p_CW[w, c])
    # print(S[c] - s)

    return S, avg_S


import itertools
def compareMaps(A, B):
    wordset_A = {a for a in A.values()}
    wordset_B = {b for b in B.values()}
    if not len(wordset_A) == len(wordset_B):
        return 0


    max_overlap = 0
    for beta in itertools.permutations(wordset_B):
        overlap = compareAssignment(A, B, wordset_A, beta)
        if overlap > max_overlap:
            max_overlap = overlap
            best_beta = beta

    return max_overlap


def compareAssignment(A, B, word_order_A, word_order_B):
    scores = []
    for word_index_a, word_index_b in zip(word_order_A, word_order_B):
        a_tiles = {t[0] for t in A.items() if t[1] == word_index_a}
        b_tiles = {t[0] for t in B.items() if t[1] == word_index_b}
        scores.append(len(b_tiles.intersection(a_tiles)) / max([len(a_tiles), len(b_tiles)]))

    return min(scores)

def sim(chip_index_x, chip_index_y, c=0.001):

    color_x = wcs.cielab_map[chip_index_x]
    color_y = wcs.cielab_map[chip_index_y]

    # CIELAB distance 76 (euclidean distance)
    d = np.linalg.norm(color_x - color_y, 2)

    # Regier color similarity
    return np.exp(-c * np.power(d, 2))

def communication_cost_regier(V, sim, sum_over_whole_s=False, norm_over_s=False, weight_by_size=False):

    s = {}
    for i in V.keys():
        s[i] = 0
        for j in V.keys():
            if V[i] == V[j]:
                s[i] += sim(i, j)


    l = {}
    for t in V.keys():
        z = 0
        cat_size = 0
        for i in V.keys():

            if sum_over_whole_s or V[i] == V[t]:
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


def wellformedness(V, sim):
    Sw = 0
    for i in V.keys():
        for j in V.keys():
            if V[i] == V[j]:
                Sw += sim(i, j)
    Da = 0
    for i in V.keys():
        for j in V.keys():
            if V[i] != V[j]:
                Da += 1- sim(i, j)
    W = Sw + Da
    return W


def compute_term_usage(V):
    def inc_dict(dict, key, increment):
        if key in dict.keys():
            dict[key] += increment
        else:
            dict[key] = increment

    cat_sizes = {}
    for v in V.values():
        inc_dict(cat_sizes, v, 1)
    n = len(cat_sizes)
    return n, cat_sizes


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

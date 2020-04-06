import itertools
import math
import torch.nn.functional as F
import torchHelpers as th

import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

import Correlation_Clustering
import torch

def agent_language_map(env, a):
    V = {}
    a = th.cuda(a)
    perception_indices, perceptions = env.full_batch()

    probs = a(perception=perceptions)
    _, terms = probs.max(1)

    for perception_index in perception_indices:
        V[perception_index] = terms[perception_index].item()
    return list(V.values())

def compute_gibson_cost2( env, a):
    _, perceptions = env.full_batch()
    perceptions = perceptions.cpu()
    p_WC = F.softmax(a(perception=perceptions), dim=1).t().data.numpy()
    return bayes_gibson(p_WC)


def bayes_gibson(p_WC):
    p_WC += np.finfo(float).eps  # Make sure there is no zero probs
    msg_dim = p_WC.shape[0]
    env_dim = p_WC.shape[1]
    # p_c = 1/env_dim # not needed since it is uniform
    z = 0
    for i in range(env_dim):
        z += p_WC[:, i]  # * p_c
    p_CW = p_WC.transpose() / z
    es = 0
    for w in range(msg_dim):
        es += -p_WC[w, :] * np.log2(p_CW[:, w])
    return es.mean()


def compute_cielab_map(wcs, k, iterations=10, bw_boost=1):
    N = wcs.data_dim()
    corr_graph = np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, i):
            corr_graph[i, j] = (wcs.sim_index(i, j, bw_boost=bw_boost).numpy() - 0.5)*100
            corr_graph[j, i] = (wcs.sim_index(i, j, bw_boost=bw_boost).numpy() - 0.5)*100
    consensus = Correlation_Clustering.max_correlation(corr_graph, k, iterations)
    return consensus


def regier2(env, map):
    if type(map) == list:
        map = np.array(map)

    env_dim = map.shape[0]
    s = np.zeros(env_dim)
    l = np.zeros(env_dim)

    for t in range(env_dim):
        partition = np.argwhere(map == map[t])[:, 0]
        #partition = partition[partition != t]
        for i in range(env_dim):
            s[i] = env.regier_reward(env.cielab_map[i], env.cielab_map[partition]).sum()
        l[t] = s[t] / s.sum()

    #l = l / l.sum()

    return -np.log2(l).sum() / env_dim


# Language map based metrics
def communication_cost_regier(env, V, sum_over_whole_s=False, norm_over_s=False, weight_by_size=False):
    s = {}
    for i in range(len(V)):
        s[i] = 0
        for j in range(len(V)):
            if V[i] == V[j]:
                s[i] += env.sim_index(i, j)

    l = {}
    for t in range(len(V)):
        z = 0
        cat_size = 0
        for i in range(len(V)):

            if sum_over_whole_s or V[i] == V[t]:
                z += s[i]
                cat_size += 1
        l[t] = s[t] / z
        if weight_by_size:
            l[t] *= cat_size / len(V)

    if norm_over_s:
        l_z = 0
        for x in l.values():
            l_z += x
        for i in l.keys():
            l[i] /= l_z


    E = 0
    for t in range(len(V)):
        E += -np.log2(l[t])
    E = E / len(V)

    return E


def wellformedness(env, V):
    Sw = 0
    Da = 0
    for i in range(len(V)):
        for j in range(i+1, len(V)):
            if V[i] == V[j]:
                Sw += env.sim_index(i, j)
            else:
                Da += 1 - env.sim_index(i, j)

    W = Sw + Da
    return W


def compute_term_usage(V):
    return np.unique(V).shape
    # def inc_dict(dict, key, increment):
    #     if key in dict.keys():
    #         dict[key] += increment
    #     else:
    #         dict[key] = increment
    #
    # cat_sizes = {}
    # for v in V:
    #     inc_dict(cat_sizes, v, 1)
    # n = len(cat_sizes)
    # return n, cat_sizes

from scipy.stats import t
def mean_rand_index(ce_a, ce_b=None):
    if ce_b is None:
        skip_trace = 1
        ce_b = ce_a
    else:
        skip_trace = 0
        # make sure ce_is the longest
        if len(ce_b) > len(ce_a):
            ce = ce_b
            ce_b = ce_a
            ce_a = ce
    ar = []
    for i in range(skip_trace, len(ce_a)):
        for j in range(0, min([i + 1, len(ce_b)]) - skip_trace):
            # debug code: print('i={},j={}'.format(i, j))
            ar += [adjusted_rand_score(ce_a[i], ce_b[j])]
    if len(ar) >= 1:
        x = np.array(ar)
        n = x.shape[-1]
        mean = x.mean()
        stddev = x.std(ddof=1)  # Sample variance

        # Get the endpoints of the range that contains 95% of the distribution
        # The degree used in calculations is N - ddof
        confidence_interval = 0.95
        t_bounds = t.interval(confidence_interval, n - 1)
        c = t_bounds[1] * stddev / np.sqrt(n)

        return mean, c, n
    else:
        return float('nan')




        return mean, ci

from scipy.stats import ttest_ind
def check_hypothesis(ce_a, ce_b, ce_c=None):
    alpha = 0.05
    x_1 = helper_hypo(ce_a, ce_c)
    x_2 = helper_hypo(ce_b, ce_c)
    # Welch's t-test
    t_stat, p_value = ttest_ind(x_1, x_2, equal_var=True, nan_policy='omit')
    return t_stat, p_value


def helper_hypo(ce_a, ce_b=None):
    if ce_b is None:
        skip_trace = 1
        ce_b = ce_a
    else:
        skip_trace = 0
        # make sure ce_is the longest
        if len(ce_b) > len(ce_a):
            ce = ce_b
            ce_b = ce_a
            ce_a = ce
    ar = []
    for i in range(skip_trace, len(ce_a)):
        for j in range(0, min([i + 1, len(ce_b)]) - skip_trace):
            # debug code: print('i={},j={}'.format(i, j))
            ar += [adjusted_rand_score(ce_a[i], ce_b[j])]
    if len(ar) >= 1:
        x = np.array(ar)
        return x
    else:
        return float('nan')
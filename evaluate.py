import itertools
import math

import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

import Correlation_Clustering


def compute_cielab_map(wcs, k, iterations=10, bw_boost=1):
    N = wcs.data_dim()
    corr_graph = np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, i):
            corr_graph[i, j] = (wcs.sim_index(i, j, bw_boost=bw_boost).numpy() - 0.5)*100
            corr_graph[j, i] = (wcs.sim_index(i, j, bw_boost=bw_boost).numpy() - 0.5)*100
    consensus = Correlation_Clustering.max_correlation(corr_graph, k, iterations)
    return consensus


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
    ar = 0
    n = 0
    for i in range(skip_trace, len(ce_a)):
        for j in range(0, min([i + 1, len(ce_b)]) - skip_trace):
            # debug code: print('i={},j={}'.format(i, j))
            ar += adjusted_rand_score(ce_a[i], ce_b[j])
            n += 1
    if n >= 1:
        return ar / n
    else:
        return float('nan')

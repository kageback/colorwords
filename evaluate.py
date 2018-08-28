import numpy as np
import itertools
import math
import Correlation_Clustering
from sklearn.metrics.cluster import adjusted_rand_score


def compute_consensus_map(cluster_ensemble, iter, k):
    N = len(cluster_ensemble[0])
    corr_graph = np.zeros((N, N))
    for ss in cluster_ensemble:
        for i in range(0, N):
            for j in range(0, i):
                if ss[i] == ss[j]:
                    corr_graph[i, j] = corr_graph[i, j] + 1
                    corr_graph[j, i] = corr_graph[i, j] + 1
                else:
                    corr_graph[i, j] = corr_graph[i, j] - 1
                    corr_graph[j, i] = corr_graph[i, j] - 1

    consensus = Correlation_Clustering.max_correlation(corr_graph, k, iter)
    return consensus


def compute_cielab_map(wcs, k, iterations=10, bw_boost=1):
    N = wcs.data_dim()
    corr_graph = np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, i):
            corr_graph[i, j] = (wcs.sim_index(i, j, bw_boost=bw_boost).numpy() - 0.5)*100
            corr_graph[j, i] = (wcs.sim_index(i, j, bw_boost=bw_boost).numpy() - 0.5)*100
    consensus = Correlation_Clustering.max_correlation(corr_graph, k, iterations)
    return consensus


def does_noise_matter_for_partitioning_style(exp):

    maps = exp.reshape('agent_language_map')
    terms_used = exp.reshape('term_usage')

    maps_per_noise = exp.reshape('agent_language_map', as_function_of_axes=['perception_noise'])
    terms_used_per_noise = exp.reshape('term_usage', as_function_of_axes=['perception_noise'])

    asdf = np.array([mean_rand_index(maps[terms_used == t])
                     for t in np.unique(terms_used)])

    zxcv = np.array([[mean_rand_index(maps_per_noise[noise_i][terms_used_per_noise[noise_i] == t])
                      for noise_i in range(len(maps_per_noise))]
                     for t in np.unique(terms_used)])
    fasd = np.array([a[~np.isnan(a)].mean() for a in zxcv])

    for i in range(len(np.unique(terms_used))):
        print('{:2d} & {:.3f} & {:.3f} & {:.3f} \\\\ \\hline'.format(np.unique(terms_used)[i],
                                                                     asdf[i],
                                                                     fasd[i],
                                                                     fasd[i] / asdf[i]))

    print('Terms used & Mean rand index for all & Mean rand index within noise group & Ratio \\\\ \\thickhline')
    number_of_terms = np.unique(terms_used)
    for t in number_of_terms:
        # compute average distance between all maps within a number of terms used
        m = maps[terms_used == t]
        mean_rand = mean_rand_index(m)
        # compute average distance between all maps within a number of terms used and perception noise used.
        v = 0
        n = 0
        for noise_i in range(len(maps_per_noise)):
            m = maps_per_noise[noise_i][terms_used_per_noise[noise_i] == t]
            a = mean_rand_index(m)
            if not math.isnan(a):
                v += a
                n += 1
        if n >= 1:
            mean_per_nois = v / n
        else:
            mean_per_nois = float('nan')
        #print('Terms used = {:2d} | Mean rand index: All = {:.3f} | Noise group = {:.3f} '
        #      '| Ratio = {:.3f} \\\\'.format(t, mean_rand, mean_per_nois, mean_per_nois / mean_rand))

        # print result as latex table
        print('{:2d} & {:.3f} & {:.3f} & {:.3f} \\\\ \\hline'.format(t, mean_rand, mean_per_nois, mean_per_nois / mean_rand))





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
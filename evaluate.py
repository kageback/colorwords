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
    #consensus = {k: consensus[k] for k in range(len(consensus))}
    return consensus


def does_noise_matter_for_partitioning_style(exp):

    maps = exp.reshape('agent_language_map')
    terms_used = exp.reshape('term_usage')

    maps_per_noise = exp.reshape('agent_language_map', as_function_of_axes=['perception_noise'])
    terms_used_per_noise = exp.reshape('term_usage', as_function_of_axes=['perception_noise'])

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

def mean_rand_index(m):
    ar = 0
    n = 0
    for i in range(len(m)):
        for j in range(0, i):
            ar += adjusted_rand_score(m[i], m[j])
            n += 1
    if n >= 1:
        mean_rand = ar / n
    else:
        mean_rand = float('nan')
    return mean_rand


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
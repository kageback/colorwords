import numpy as np
import itertools

import Correlation_Clustering


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
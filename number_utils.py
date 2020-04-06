import itertools
import math
import torch.nn.functional as F
import torchHelpers as th
import sys
import random
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

import Correlation_Clustering
import torch
import matplotlib.pyplot as plt


def reciever_probs(V, reciever):
    receiver = th.cuda(reciever)
    words = list(set(V))
    msg = torch.LongTensor(np.asarray(words, dtype=int)).unsqueeze(1)
    # Why is this needed?
    msg.data = msg.data.squeeze(1)
    guess_logits = reciever(msg=msg)
    guess_probs = F.softmax(guess_logits,dim=1)
    guess_probs = guess_probs.detach().cpu().numpy()
    probs = {words[i] : guess_probs[i, :] for i in range(len(words))}
    # argmax
    sample = guess_logits.argmax(dim=1).detach().cpu().numpy() + 1
    guess = {words[i] : sample[i] for i in range(len(words))}
    print(guess)
    return probs, guess

def fraction_response(env, sender):
    sender = th.cuda(sender)
    perception_indices, perceptions = env.full_batch()
    msg_logits = sender(perception=perceptions)
    msg_probs = F.softmax(msg_logits, dim=1).detach().cpu().numpy()
    return msg_probs

def sort_fractions(exp):
    fractions = exp.get_result('sender_fraction_response')
    fractions = get_wrappers(fractions)
    msg_dim = list(exp.param_ranges['msg_dim'])
    # Sort the experiments by msg_dim
    msg_dict = {}
    for e in fractions:
        msg_probs = e.get()
        n_words = msg_probs.shape[1]
        if not str(n_words) in list(msg_dict.keys()):
            msg_dict[str(n_words)] = [msg_probs]
        else:
            msg_dict[str(n_words)] += [msg_probs]
    return msg_dict


def get_wrappers(nested_list):
    flatten_list = []
    for i in range(len(nested_list)):
        if not isinstance(nested_list[i], list):
            flatten_list.append(nested_list[i])
        else:
            flatten_list += get_wrappers(nested_list[i])
    return flatten_list


# Correlation Clustering
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
    consensus = max_correlation(corr_graph, k, iter)
    return consensus


def max_correlation(my_graph, my_K, my_itr_num):

    my_N = np.size(my_graph,0)

    best_Obj = -sys.float_info.max
    best_Sol = np.zeros((my_N,), dtype = int)

    for itr in range(0,my_itr_num):

        cur_Sol = np.zeros((my_N,), dtype = int)

        for i in range(0,my_N):
            cur_Sol[i] = random.randint(0,my_K-1)

        # calc total cost
        cur_Obj = 0.0
        for i in range(0,my_N):
            for j in range(0,i):
                if cur_Sol[i] == cur_Sol[j]:
                    cur_Obj = cur_Obj + my_graph[i,j]


        old_Obj = cur_Obj - 1

        while cur_Obj-old_Obj > sys.float_info.epsilon:
            old_Obj = cur_Obj

            order = list(range(0,my_N))
            random.shuffle(order)

            for i in range(0,my_N):
                cur_Ind = order[i]
                temp_Objs = np.zeros((my_K,), dtype = float)

                for j in range(0,my_N):
                    if j != cur_Ind:
                        temp_Objs[cur_Sol[j]] = temp_Objs[cur_Sol[j]] + my_graph[cur_Ind,j]


                sep_Obj = temp_Objs[cur_Sol[cur_Ind]]
                temp_Objs[cur_Sol[cur_Ind]] = cur_Obj

                for k in range(0,my_K):
                    if k != cur_Sol[cur_Ind]:
                        temp_Objs[k] = cur_Obj - sep_Obj + temp_Objs[k]


                temp_max = np.argmax(temp_Objs)
                cur_Sol[cur_Ind] = temp_max
                cur_Obj = temp_Objs[temp_max]

        if itr == 0 or cur_Obj > best_Obj:
            best_Sol = np.array(cur_Sol)
            best_Obj = cur_Obj

    return best_Sol

def plot_map(agent_consensus_map, save_to_path):
    words = np.unique(agent_consensus_map)
    partition = compute_partition(agent_consensus_map)
    for word in words:
        mask = agent_consensus_map == word
        numbers = np.argwhere(agent_consensus_map == word) + 1
        plt.plot(numbers, np.ones(numbers.shape) + word)
    plt.savefig(save_to_path)

#def compute_partition(agent_consensus_map):

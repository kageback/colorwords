import scipy.optimize as opt
import numpy as np
from numba import jit
# Objective min sum_n p(n) * abs(n - w_n^T * q)
# Constraint to :
#   Sender :
#       for all n and all words j, w_nj in {0, 1} such that sum_j w_nj = 1
#   Reciever :
#       for all words j we have q_j such that q_j in N
# p(n) is the prior

# Smoothing?

# decision variables
# x in R^((N+1) * W)
# A = x[0:N * W].reshape(N, W) sender
# q = x[N * W : N * W + W] Reciever
N = 2
W = 2
prior = np.load('data/ngram.npy')
prior = prior[0:N] / np.sum(prior[0:N])
def objective(x):
    A = x[0: N*W].reshape(N, W)
    q = x[N*W: (N + 1) * W]
    return  np.sum((prior * ((np.arange(N) + 1) - np.dot(A, q).flatten())**2.0))

# Constraints
# Relaxation
constraints = []
bounds = []
#   Sender
for i in range(N*W):
    constraints.append({'type': 'ineq', 'fun': lambda x: x[i]})
    constraints.append({'type': 'ineq', 'fun': lambda x: 1 - x[i]})

for i in range(W):
    constraints.append({'type': 'ineq', 'fun': lambda x: 1 - np.sum(x[i*N: (i+1)*N])})
    constraints.append({'type': 'ineq', 'fun': lambda x: np.sum(x[i*N: (i+1)*N]) - 1})

#   Reciever
for i in range(W):
    constraints.append({'type': 'ineq', 'fun': lambda x: x[N * W + i] - 1})
# bounds
# for i in range(N*W):
#     bounds.append((0, 1))
# for i in range(N*W, (N+1)*W):
#     bounds.append((1, None))

x_init = np.random.uniform(low=0, high=N, size=(N + 1) * W)
for i in range(W):
    x_init[i*N: (i + 1) * N] = x_init[i*N: (i + 1) * N] / np.sum(x_init[i*N: (i + 1) * N])
    x_init[N * W + i] = np.random.uniform(low=0, high=N)
res = opt.minimize(objective, x0=x_init, constraints=constraints, method='COBYLA')
print(res)
#objective(x_init)



# def r(A, q, N):
#     return (1 - 1/N * np.sum((prior *((np.arange(N) + 1) - np.dot(A, q).flatten())**2)))
#
# def find_partition(W, N):
#     best_reward = 0
#     best_partition = 0
#     A = []
#     for word in range(W):
#         tmp = np.zeros([1, W])
#         tmp[0, word] = 1
#         A.append(tmp)
#     partitions = compute_partitions(A, N)
#     guesses = compute_guesses(W, N)
#
#     for partition in partitions:
#         for guess in guesses:
#             reward = r(partition, guess, N)
#             if reward > best_reward:
#                 best_reward = reward
#                 best_partition =partition
#                 best_guess = guess
#     A = best_partition
#     q = best_guess
#     print(best_reward)
#
#
# def compute_partitions(A, N):
#     if N == 1:
#         return A
#     else:
#         # compute all permutations
#         B = compute_partitions(A, N-1)
#         partition = []
#         for a in A:
#             for b in B:
#                 partition.append(np.concatenate((a, b),axis=0))
#         return partition
#
#
# def compute_guesses(W, N):
#     if W == 1:
#         guesses = []
#         for i in range(1, N + 1):
#             tmp = np.ones([1, 1]) * i
#             guesses.append(tmp)
#     else:
#         guesses = []
#         B = compute_guesses(W - 1, N)
#         for i in range(1, N + 1):
#             tmp = np.ones([1, 1]) * i
#             for e in B:
#                 guesses.append(np.concatenate((tmp, e),axis=0))
#     return guesses
#
#
# numbers = 3
# n_words = 3
# prior = np.load('data/ngram.npy')
# prior = prior[0:numbers] / np.sum(prior[0:numbers])
# find_partition(W=n_words,N=numbers)

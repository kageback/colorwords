import numpy as np
import torchHelpers as th
import torch.nn.functional as F

class BaseEnviroment:
    def __init__(self) -> None:
        super().__init__()

    def full_batch(self):
        pass

    def mini_batch(self, batch_size=10):
        pass

    def data_dim(self):
        pass

    # Language map based metrics
    def agent_language_map(self, a):
        V = {}
        a = th.cuda(a)
        perception_indices, perceptions = self.full_batch()
        perceptions = th.float_var(perceptions)

        probs = a(perception=perceptions)
        _, terms = probs.max(1)

        for perception_index in perception_indices:
            V[perception_index] = terms[perception_index].cpu().data[0]

        return V

    def sim_np(self):
        pass

    def communication_cost_regier(self, V, sum_over_whole_s=False, norm_over_s=False, weight_by_size=False):

        s = {}
        for i in V.keys():
            s[i] = 0
            for j in V.keys():
                if V[i] == V[j]:
                    s[i] += self.sim_np(i, j)

        l = {}
        for t in V.keys():
            z = 0
            cat_size = 0
            for i in V.keys():

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

        # debug code to check it l sums to one
        l_z = 0
        for x in l.values():
            l_z += x

        E = 0
        for t in V.keys():
            E += -np.log2(l[t])
        E = E / len(V)

        return E

    def wellformedness(self, V):
        Sw = 0
        for i in V.keys():
            for j in V.keys():
                if V[i] == V[j]:
                    Sw += self.sim_np(i, j)
        Da = 0
        for i in V.keys():
            for j in V.keys():
                if V[i] != V[j]:
                    Da += 1 - self.sim_np(i, j)
        W = Sw + Da
        return W

    def compute_term_usage(self, V):
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

    # other metrics
    def compute_gibson_cost(self, a):
        _, perceptions = self.full_batch()
        perceptions = th.float_var(perceptions, False)
        all_terms = th.long_var(range(a.msg_dim), False)

        p_WC = a(perception=perceptions).t().data.numpy()
        p_CW = F.softmax(a(msg=all_terms), dim=1).data.numpy()

        S = -np.diag(np.matmul(p_WC.transpose(), (np.log2(p_CW))))
        avg_S = S.sum() / len(S)  # expectation assuming uniform prior
        # debug code
        # s = 0
        # c = 43
        # for w in range(a.msg_dim):
        #     s += -p_WC[w, c]*np.log2(p_CW[w, c])
        # print(S[c] - s)
        return S, avg_S


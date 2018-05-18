import torch
import torch.nn.functional as F
import numpy as np

import wcs
import torchHelpers as th

def compute_gibson_cost(a):
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
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
from torch.distributions import Normal

import agents
import wcs
import torchHelpers as th


# basic color chip similarity

def dist(color_x, color_y):
    # CIELAB distance 76 (euclidean distance)
    diff = (color_x - color_y)
    return diff.norm(2, 1)


def sim(color_x, color_y, c = 0.001):
    # Regier similarity
    return torch.exp(-c * torch.pow(dist(color_x, color_y), 2))


# Reward functions

def basic_reward(color_codes, color_guess):
    _, I = color_guess.max(1)
    reward = (color_codes == I).float() - (color_codes != I).float()
    return reward


def regier_reward(data, color, color_guess):
    _, color_code_guess = color_guess.max(1)
    color_guess = th.float_var(data.code2color(color_code_guess), args.cuda)
    return sim(color, color_guess)


# Evaluation

def evaluate(a, data):
    color_codes, cnum, colors = data.all_colors()
    colors = th.float_var(colors, args.cuda)
    color_codes = th.long_var(color_codes, args.cuda)

    probs = a(perception=colors)
    _, msg = probs.max(1)

    data.print(lambda t: str(
        msg[
            np.where(
                cnum == t['#cnum'].values[0]
            )[0][0]
        ].cpu().data.numpy()[0]
    ), pad=2)

    nwords = np.array([((msg == w).sum()>0).cpu().data.numpy()[0] for w in range(a.msg_dim)]).sum()
    print('word usage count', [(msg == w).sum().cpu().data.numpy()[0] for w in range(a.msg_dim)], 'num of words used', nwords)

    color_guess = a(msg=msg)
    cost, perplexity = uninformed_commcost(color_guess, color_codes)
    print("comcost: %f Perplexity %f" % (cost, perplexity))


def uninformed_commcost(color_guess, color_codes):
    y = F.softmax(color_guess, dim=1)
    ny = y.cpu().data.numpy()
    nlabes = color_codes.data.cpu().numpy()

    cost = 0
    perplexity = 0
    for t, i in zip(nlabes, range(nlabes.size)):
        cost += -np.log2(ny[i, t])
        perplexity += np.exp2(-np.log2(ny[i, t]))

    cost = cost / nlabes.size
    perplexity = perplexity / nlabes.size

    return cost, perplexity


def regier_cost(a, data):
    print('Deprecated')
    color_codes, cnums, colors = data.all_colors()
    colors = th.float_var(colors, args.cuda)
    color_codes = th.long_var(color_codes, args.cuda)

    probs = a(perception=colors)
    _, w_map = probs.max(1)

    s = th.float_var(torch.zeros(len(color_codes)), args.cuda)
    for t_color_code, t_cnum, t_color in zip(color_codes, cnums, colors):
        w = w_map[t_color_code]
        cat_colors = th.float_var(data.code2color(color_codes[w_map == w]), args.cuda)
        s[t_color_code.data[0]] = sim(t_color, cat_colors).sum().data[0]

    l = th.float_var(torch.zeros(len(color_codes)), args.cuda)
    for t_color_code, t_cnum, t_color in zip(color_codes, cnums, colors):
        w = w_map[t_color_code]
        l[t_color_code.data[0]] = s[t_color_code.data[0]] / s[color_codes[w_map == w]].sum().data[0]

    E = -np.log2(l.data.cpu().numpy()).mean()
    return E


def color_graph_V(a, data, to_numpy=True):
    V = {}

    color_codes, cnums, colors = data.all_colors()
    colors = th.float_var(colors, args.cuda)
    color_codes = th.long_var(color_codes, args.cuda)

    probs = a(perception=colors)
    _, words = probs.max(1)

    for color_code, cnum, color in zip(color_codes, cnums, colors):
        if to_numpy:
            V[cnum] = {'word': words[color_code].cpu().data[0],
                       'CIELAB_color': color.cpu().data.numpy()}
        else:
            V[cnum] = {'word': words[color_code], 'CIELAB_color': color}

    return V


def communication_cost_regier(V, sum_over_whole_s=False):

    s = {}
    for i in V.keys():
        s[i] = 0
        for j in V.keys():
            if V[i]['word'] == V[j]['word']:
                s[i] += sim_numpy(V[i]['CIELAB_color'], V[j]['CIELAB_color'])


    l = {}
    for t in V.keys():
        z = 0
        for i in V.keys():
            if sum_over_whole_s or V[i]['word'] == V[t]['word']:
                z += s[i]
        l[t] = s[t]/z

    E = 0
    for t in V.keys():
        E += -np.log2(l[t])
    E = E / len(V)

    return E

def xrange(start,stop):
    return range(start,stop+1)

def min_k_cut_cost(V,k):

    C = {}
    for i in xrange(1, k):
        C[i] = []

    for cnum in V.keys():
        C[ V[cnum]['word']+1 ].append(cnum)

    cost = 0
    for i in xrange(1, k-1):
        for j in xrange(i+1, k):
            for v1 in C[i]:
                for v2 in C[j]:
                    cost += sim_numpy(V[v1]['CIELAB_color'], V[v2]['CIELAB_color'])
    return cost



def sim_numpy(color_x, color_y, c=0.001):

    # CIELAB distance 76 (euclidean distance)
    d = np.linalg.norm(color_x - color_y, 2)

    # Regier color similarity
    return np.exp(-c * np.power(d, 2))



# Model training loop

def main(args,
         perception_dim=3,
         reward_func='regier_reward',
         print_wcs_cnum_map=False):

    data = wcs.WCSColorData()

    if print_wcs_cnum_map:
        data.print(lambda t: str(t['#cnum'].values[0]), pad=4)

    a = th.cuda(agents.BasicAgent(args.msg_dim, args.hidden_dim, data.color_dim, perception_dim), args.cuda)

    optimizer = optim.Adam(a.parameters())
    criterion_receiver = torch.nn.CrossEntropyLoss()

    sumrev = 0
    for i in range(args.max_epochs):
        optimizer.zero_grad()

        color_codes, colors = data.batch(batch_size=args.batch_size)
        color_codes = th.long_var(color_codes, args.cuda)
        colors = th.float_var(colors, args.cuda)

        noise = th.float_var(Normal(torch.zeros(args.batch_size, perception_dim),
                                    torch.ones(args.batch_size, perception_dim) * args.noise_level).sample(), args.cuda)
        colors = colors + noise

        probs = a(perception=colors)
        m = Categorical(probs)
        msg = m.sample()

        color_guess = a(msg=msg)

        loss_receiver = criterion_receiver(color_guess, color_codes)

        if reward_func == 'basic_reward':
            reward = basic_reward(color_codes, color_guess)
        elif reward_func == 'regier_reward':
            reward = regier_reward(data, colors, color_guess)

        loss_sender = args.sender_loss_multiplier * ((-m.log_prob(msg) * reward).sum()/args.batch_size)

        loss = loss_receiver + loss_sender
        loss.backward()

        optimizer.step()

        # printing status and periodic evaluation
        sumrev += reward.sum()

        if i % 100 == 0:
            print("Loss sender %f Loss receiver %f Naive perplexity %f Average reward %f Min k-cut cost %f" %
                  (loss_sender,
                   loss_receiver,
                   torch.exp(loss_receiver), sumrev / (args.print_interval*args.batch_size),
                   min_k_cut_cost(color_graph_V(a, data), a.msg_dim))
                  )
            sumrev = 0

            #debug
            #cut = min_k_cut_cost(color_graph_V(a, data), a.msg_dim)
            #r2 = communication_cost_regier(color_graph_V(a, data))
            #r_old = regier_cost(a, data)

        if args.periodic_evaluation != 0 and (i % args.periodic_evaluation == 0):
            evaluate(a, data)
            #print('Regier cost: %f' % (regier_cost(a, data)))

    return color_graph_V(a, data)


# Script entry point

if __name__ == "__main__":
    import pickle
    import argparse

    parser = argparse.ArgumentParser(description='Agents learning to communicate color')
    #parser.add_argument('--cuda', action='store_true',
    #                    help='use CUDA')
    parser.add_argument('--save_path', type=str, default='save',
                        help='path for saving logs and results')
    parser.add_argument('--exp_name', type=str, default='dev',
                        help='path for saving logs and results')
    parser.add_argument('--msg_dim', type=int, default=2,
                        help='Number of color words')
    parser.add_argument('--max_epochs', type=int, default=10000,
                        help='Number of training epochs')
    parser.add_argument('--noise_level', type=int, default=5,
                        help='How much noise to add to the color chips')
    parser.add_argument('--hidden_dim', type=int, default=20,
                        help='size of hidden representation')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of epochs to run in parallel before updating parameters')
    parser.add_argument('--sender_loss_multiplier', type=int, default=100,
                        help='Mixing factor when mixing the sender with the receiver part of the objective')
    parser.add_argument('--print_interval', type=int, default=200,
                        help='How often to print training state')
    parser.add_argument('--periodic_evaluation', type=int, default=500,
                        help='How often to perform periodic evaluation. Set 0 for no periodic evaluation.')

    args = parser.parse_args()
    print(args)

    #if args.cuda:
    args.cuda = torch.cuda.is_available()

    res = {}
    res['V'] = main(args)
    res['regier_cost'] = communication_cost_regier(res['V'])

    with open(args.save_path + '/' + args.exp_name + '.result.pkl', 'wb') as f:
        pickle.dump(res, f)


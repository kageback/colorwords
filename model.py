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


def regier_reward(color, color_guess):
    _, color_code_guess = color_guess.max(1)
    color_guess = th.float_var(wcs.chip_index2CIELAB(color_code_guess.data), args.cuda)
    return sim(color, color_guess)


# Evaluation

def evaluate(a):
    chip_indices, colors = wcs.all_colors()
    colors = th.float_var(colors, args.cuda)

    probs = a(perception=colors)
    _, msg = probs.max(1)

    wcs.print_color_map(lambda t: str(
        msg[
            np.where(
                chip_indices == t.index.values[0]
            )[0][0]
        ].cpu().data.numpy()[0]
    ), pad=2)

    nwords = np.array([((msg == w).sum()>0).cpu().data.numpy()[0] for w in range(a.msg_dim)]).sum()
    print('word usage count', [(msg == w).sum().cpu().data.numpy()[0] for w in range(a.msg_dim)], 'num of words used', nwords)

    color_guess = a(msg=msg)
    cost, perplexity = uninformed_commcost_log2(color_guess, th.long_var(chip_indices, args.cuda))
    print("comcost: %f Perplexity %f" % (cost, perplexity))


def uninformed_commcost_log2(chip_index_guess, chip_indices):
    y = F.softmax(chip_index_guess, dim=1)
    ny = y.cpu().data.numpy()
    nlabes = chip_indices.data.cpu().numpy()

    cost = 0
    perplexity = 0
    for t, i in zip(nlabes, range(nlabes.size)):
        cost += -np.log2(ny[i, t])
        perplexity += np.exp2(-np.log2(ny[i, t]))

    cost = cost / nlabes.size
    perplexity = perplexity / nlabes.size

    return cost, perplexity


def color_graph_V(a):
    V = {}

    chip_indices, colors = wcs.all_colors()
    colors = th.float_var(colors, args.cuda)
    #chip_indices = th.long_var(chip_indices, args.cuda)

    probs = a(perception=colors)
    _, words = probs.max(1)

    for chip_index in chip_indices:
        V[chip_index] = {'word': words[chip_index].cpu().data[0]}

    return V


# Model training loop

def main(args,
         perception_dim=3,
         reward_func='regier_reward',
         print_wcs_cnum_map=False):

    if print_wcs_cnum_map:
        wcs.print_color_map(lambda t: str(t['#cnum'].values[0]), pad=4)

    a = th.cuda(agents.BasicAgent(args.msg_dim, args.hidden_dim, wcs.color_dim(), perception_dim), args.cuda)

    optimizer = optim.Adam(a.parameters())
    criterion_receiver = torch.nn.CrossEntropyLoss()

    sumrev = 0
    for i in range(args.max_epochs):
        optimizer.zero_grad()

        color_codes, colors = wcs.batch(batch_size=args.batch_size)
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
            reward = regier_reward( colors, color_guess)

        loss_sender = args.sender_loss_multiplier * ((-m.log_prob(msg) * reward).sum()/args.batch_size)

        loss = loss_receiver + loss_sender
        loss.backward()

        optimizer.step()

        # printing status and periodic evaluation
        sumrev += reward.sum()

        if args.print_interval != 0 and (i % args.print_interval == 0):
            V = color_graph_V(a)
            print("Loss sender %f Loss receiver %f Naive perplexity %f Average reward %f Min k-cut cost %f Regier_commcost %f" %
                  (loss_sender,
                   loss_receiver,
                   torch.exp(loss_receiver), sumrev / (args.print_interval*args.batch_size),
                   wcs.min_k_cut_cost(V, a.msg_dim),
                   wcs.communication_cost_regier(V,sum_over_whole_s=False))
                  )
            sumrev = 0

            #debug
            #cut = min_k_cut_cost(color_graph_V(a), a.msg_dim)
            #r2 = communication_cost_regier(color_graph_V(a))
            #r_old = regier_cost(a)

        if args.periodic_evaluation != 0 and (i % args.periodic_evaluation == 0):
            evaluate(a)
            #print('Regier cost: %f' % (regier_cost(a)))

    return color_graph_V(a)


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
    parser.add_argument('--msg_dim', type=int, default=11,
                        help='Number of color words')
    parser.add_argument('--max_epochs', type=int, default=10000,
                        help='Number of training epochs')
    parser.add_argument('--noise_level', type=int, default=0,
                        help='How much noise to add to the color chips')
    parser.add_argument('--hidden_dim', type=int, default=20,
                        help='size of hidden representation')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of epochs to run in parallel before updating parameters')
    parser.add_argument('--sender_loss_multiplier', type=int, default=100,
                        help='Mixing factor when mixing the sender with the receiver part of the objective')
    parser.add_argument('--print_interval', type=int, default=5000,
                        help='How often to print training state.  Set 0 for no printing')
    parser.add_argument('--periodic_evaluation', type=int, default=0,
                        help='How often to perform periodic evaluation. Set 0 for no periodic evaluation.')

    args = parser.parse_args()
    print(args)

    #if args.cuda:
    args.cuda = torch.cuda.is_available()

    res = {}
    res['args'] = args
    res['V'] = main(args)

    # Extras: As these are compact and takes some computing I will add them to the result even though they can be computed later based on V
    res['regier_cost'] = wcs.communication_cost_regier(res['V'])
    res['regier_cost'] = wcs.min_k_cut_cost(res['V'], res['args'].msg_dim)
    # ==========

    with open(args.save_path + '/' + args.exp_name + '.result.pkl', 'wb') as f:
        pickle.dump(res, f)


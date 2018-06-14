import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
from torch.distributions import Normal

import agents
import torchHelpers as th
import com_enviroments




# Evaluation

def evaluate(a,cuda):
    chip_indices, colors = wcs.all_colors()
    colors = th.float_var(colors, cuda)

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
    cost, perplexity = uninformed_commcost_log2(color_guess, th.long_var(chip_indices, cuda))
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


def color_graph_V(a, wcs, cuda=torch.cuda.is_available()):
    V = {}

    chip_indices, colors = wcs.all_colors()
    colors = th.float_var(colors, cuda)
    #chip_indices = th.long_var(chip_indices, cuda)

    probs = a(perception=colors)
    _, words = probs.max(1)

    for chip_index in chip_indices:
        V[chip_index] = words[chip_index].cpu().data[0]

    return V


# Model training loop

def main(cuda=torch.cuda.is_available(),
         task='wcs',
         com_model='onehot',
         com_noise=0,
         msg_dim=11,
         max_epochs=1000,
         noise_level=0,
         hidden_dim=20,
         batch_size=100,
         sender_loss_multiplier=100,
         print_interval=1000,
         eval_interlval=0,
         perception_dim=3,
         reward_func='regier_reward'):

    wcs = com_enviroments.make(task)

    if com_model == 'onehot':
        a = th.cuda(agents.BasicAgent(msg_dim, hidden_dim, wcs.color_dim(), perception_dim), cuda)
    elif com_model == 'softmax':
        a = th.cuda(agents.SoftmaxAgent(msg_dim, hidden_dim, wcs.color_dim(), perception_dim), cuda)

    optimizer = optim.Adam(a.parameters())
    criterion_receiver = torch.nn.CrossEntropyLoss()

    sumrev = 0
    for i in range(max_epochs):
        optimizer.zero_grad()

        color_codes, colors = wcs.batch(batch_size=batch_size)
        color_codes = th.long_var(color_codes, cuda)
        colors = th.float_var(colors, cuda)

        noise = th.float_var(Normal(torch.zeros(batch_size, perception_dim),
                                    torch.ones(batch_size, perception_dim) * noise_level).sample(), cuda)
        colors = colors + noise

        probs = a(perception=colors)

        if com_model == 'onehot':
            m = Categorical(probs)
            msg = m.sample()

            color_guess = a(msg=msg)

            if reward_func == 'basic_reward':
                reward = wcs.basic_reward(color_codes, color_guess)
            elif reward_func == 'regier_reward':
                reward = wcs.regier_reward(colors, color_guess, cuda)

            sumrev += reward.sum()

            loss_sender = sender_loss_multiplier * ((-m.log_prob(msg) * reward).sum()/batch_size)

        elif com_model == 'softmax':
            loss_sender = 0
            noise = th.float_var(Normal(torch.zeros(batch_size, msg_dim),
                                        torch.ones(batch_size, msg_dim) * com_noise).sample(), cuda)
            msg=probs+noise
            color_guess = a(msg=msg)


        loss_receiver = criterion_receiver(color_guess, color_codes)
        loss = loss_receiver + loss_sender

        loss.backward()

        optimizer.step()

        # printing status and periodic evaluation

        if print_interval != 0 and (i % print_interval == 0):
            print("Loss sender %f Loss receiver %f Naive perplexity %f Average reward %f" %
                  (loss_sender,
                   loss_receiver,
                   torch.exp(loss_receiver), sumrev / (print_interval*batch_size))
                  )
            sumrev = 0


        if eval_interlval != 0 and (i % eval_interlval == 0):
            evaluate(a)
            V = color_graph_V(a)
            print("Min k-cut cost %f Regier_commcost %f" %
                  (wcs.min_k_cut_cost(V, a.msg_dim),
                   wcs.communication_cost_regier(V))
                  )

    return a.cpu()


# Script entry point
if __name__ == "__main__":
    agent = main()
    # outdated code kept for ref until lift to new GE done
    # import pickle
    # import argparse
    #
    # parser = argparse.ArgumentParser(description='Agents learning to communicate color')
    # parser.add_argument('--save_path', type=str, default='save',
    #                     help='path for saving logs and results')
    # parser.add_argument('--exp_name', type=str, default='dev',
    #                     help='path for saving logs and results')
    # parser.add_argument('--msg_dim', type=int, default=11,
    #                     help='Number of color words')
    # parser.add_argument('--max_epochs', type=int, default=1000,
    #                     help='Number of training epochs')
    # parser.add_argument('--noise_level', type=int, default=0,
    #                     help='How much noise to add to the color chips')
    # parser.add_argument('--hidden_dim', type=int, default=20,
    #                     help='size of hidden representation')
    # parser.add_argument('--batch_size', type=int, default=100,
    #                     help='Number of epochs to run in parallel before updating parameters')
    # parser.add_argument('--sender_loss_multiplier', type=int, default=100,
    #                     help='Mixing factor when mixing the sender with the receiver part of the objective')
    # parser.add_argument('--print_interval', type=int, default=1000,
    #                     help='How often to print training state.  Set 0 for no printing')
    # parser.add_argument('--periodic_evaluation', type=int, default=0,
    #                     help='How often to perform periodic evaluation. Set 0 for no periodic evaluation.')
    #
    # args = parser.parse_args()
    # print(args)
    #
    # #if args.cuda:
    # args.cuda = torch.cuda.is_available()
    #
    # res = {}
    # res['args'] = args
    # V, agent = main(args)
    # res['V'] = V
    # res['agent'] = agent
    #
    # with open(args.save_path + '/' + args.exp_name + '.result.pkl', 'wb') as f:
    #     pickle.dump(res, f)


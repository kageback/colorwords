import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
from torch.distributions import Normal

import agents
import wcs
import torchHelpers as th


def basic_reward(color_codes, color_guess):
    _, I = color_guess.max(1)
    reward = (color_codes == I).float() - (color_codes != I).float()
    return reward


def regier_reward(data, color, color_guess):
    _, color_code_guess = color_guess.max(1)
    color_guess = th.float_var(data.code2color(color_code_guess))
    return sim(color, color_guess)

def regier_cost(a, data):

    color_codes, cnums, colors = data.all_colors()
    colors = th.float_var(colors)
    color_codes = th.long_var(color_codes)

    probs = a(perception=colors)
    _, w_map = probs.max(1)



    s = th.float_var(torch.zeros(len(color_codes)))
    for t_color_code, t_cnum, t_color in zip(color_codes, cnums, colors):
        w = w_map[t_color_code]
        cat_colors = th.float_var(data.code2color(color_codes[w_map == w]))
        s[t_color_code.data[0]] = sim(t_color, cat_colors).sum().data[0]

    l = th.float_var(torch.zeros(len(color_codes)))
    for t_color_code, t_cnum, t_color in zip(color_codes, cnums, colors):
        w = w_map[t_color_code]
        #cat_colors = th.float_var(data.code2color(color_codes[w_map == w]))

        l[t_color_code.data[0]] = s[t_color_code.data[0]] / s[color_codes[w_map == w]].sum().data[0]

    E = -np.log2(l.data.cpu().numpy()).mean()
    return E

def sim(color_x, color_y, c = 0.001):
    # Regier similarity
    return torch.exp(-c * torch.pow(dist(color_x, color_y), 2))


def dist(color_x, color_y):
    # CIELAB distance 76 (euclidean distance)
    diff = (color_x - color_y)
    return diff.norm(2, 1)


def evaluate(a, data):
    color_codes, cnum, colors = data.all_colors()
    colors = th.float_var(colors)
    color_codes = th.long_var(color_codes)

    probs = a(perception=colors)
    _, msg = probs.max(1)

    def word_used_per_color(t):
        w_index = np.where(cnum == t['#cnum'].values[0])[0][0]
        return str(msg[w_index].cpu().data.numpy()[0])
    data.print(word_used_per_color, pad=2)

    nwords = np.array([((msg == w).sum()>0).cpu().data.numpy()[0] for w in range(a.msg_dim)]).sum()
    print('word usage count', [(msg == w).sum().cpu().data.numpy()[0] for w in range(a.msg_dim)], 'num of words used', nwords)

    color_guess = a(msg=msg)
    cost, perplexity = communicative_cost(color_guess, color_codes)
    print("comcost: %f Perplexity %f" % (cost, perplexity))


def communicative_cost(color_guess, color_codes):
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


def main(
    perception_dim=3,
    limit_colors=0,
    msg_dim=11,
    noise_level=0,  # 30
    hidden_dim=20,
    batch_size=100,
    start_tau=1,
    max_epochs=10000,
    reward_func='regier_reward',
    eval=True):




    data = wcs.WCSColorData(limit_colors=limit_colors)
    if eval:
        data.print(lambda t: str(t['#cnum'].values[0]),pad=4)

    a = agents.BasicAgent(msg_dim, hidden_dim, data.color_dim, perception_dim).cuda()
    optimizer = optim.Adam(a.parameters())
    criterion_receiver = torch.nn.CrossEntropyLoss()
    sumrev = 0
    for i in range(1,max_epochs):
        optimizer.zero_grad()

        color_codes, colors = data.batch(batch_size=batch_size)
        color_codes = th.long_var(color_codes)
        colors = th.float_var(colors)

        noise = th.float_var(Normal(torch.zeros(batch_size, perception_dim),
                                    torch.ones(batch_size, perception_dim) * noise_level).sample())
        colors = colors + noise

        tau = start_tau/(i)
        tau = tau if tau > 1 else 1

        probs = a(perception=colors, tau=tau)
        m = Categorical(probs)
        msg = m.sample()

        color_guess = a(msg=msg)

        loss_receiver = criterion_receiver(color_guess, color_codes)

        if reward_func == 'basic_reward':
            reward = basic_reward(color_codes, color_guess)
        elif reward_func == 'regier_reward':
            reward = regier_reward(data, colors, color_guess)

        sumrev += reward.sum()
        loss_sender = -m.log_prob(msg) * reward
        loss_sender = loss_sender.sum()

        loss = loss_receiver + loss_sender
        loss.backward()

        optimizer.step()

        if i % 100 == 0:
            print("error %f perplex %f reward %f avg %f tau %f" % (loss_receiver, torch.exp(loss_receiver), reward.sum() / batch_size, sumrev.cpu().data.numpy() / (100*batch_size), tau ))
            sumrev = 0

        if eval and (i % 5000 == 0):
            evaluate(a, data)
            print('Regier cost: %f' % (regier_cost(a, data)))

    return regier_cost(a, data)

if __name__ == "__main__":
    main()

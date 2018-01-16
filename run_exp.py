import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.distributions import Normal

import wcs
import agents

perception_dim = 3
limit_colors = 0

msg_dim = 7
noise_level = 40

hidden_dim = 20

batch_size = 1000
start_tau = 1

def print_language(a, data):

    color_codes, cnum, colors = data.all_colors()
    #color_codes = Variable(torch.from_numpy(color_codes)).cuda()
    colors = Variable(torch.FloatTensor(colors)).cuda()
    probs = a(perception=colors)
    _, I = probs.max(1)

    def word_used_per_color(t):
        w_index = np.where(cnum == t['#cnum'].values[0])[0][0]
        return str(I[w_index].cpu().data.numpy()[0])

    data.print(word_used_per_color)
    nwords = np.array([((I == w).sum()>0).cpu().data.numpy()[0] for w in range(msg_dim)]).sum()
    print('word usage count', [(I == w).sum().cpu().data.numpy()[0] for w in range(msg_dim)], 'num of words used', nwords)


def print_guess(t):
    return str(t['#cnum'].values[0])

def main():
    data = wcs.WCSColorData(limit_colors=limit_colors)
    data.print(lambda t: str(t['#cnum'].values[0]),pad=4)

    a = agents.BasicAgent(msg_dim, hidden_dim, data.color_dim, perception_dim).cuda()
    optimizer = optim.Adam(a.parameters())
    criterion_receiver = torch.nn.CrossEntropyLoss()
    sumrev = 0
    for i in range(1,10000000):
        optimizer.zero_grad()

        color_codes, colors = data.batch(batch_size=batch_size)
        color_codes = Variable(torch.from_numpy(color_codes)).cuda()
        noise = Normal(torch.zeros(batch_size,perception_dim), torch.ones(batch_size,perception_dim) * noise_level).sample()
        colors = Variable(torch.FloatTensor(colors) + noise).cuda()




        tau = start_tau/(i)
        tau = tau if tau > 1 else 1

        probs = a(perception=colors, tau=tau)
        m = Categorical(probs)
        msg = m.sample()

        color_guess = a(msg=msg)

        loss_receiver = criterion_receiver(color_guess, color_codes)

        _, I = color_guess.max(1)
        reward = (color_codes == I).float() - (color_codes != I).float()
        sumrev += reward.sum()
        loss_sender = -m.log_prob(msg) * reward
        loss_sender = loss_sender.sum()

        loss = loss_receiver + loss_sender
        loss.backward()

        optimizer.step()

        if i % 100 == 0:
            print("error %f reward %f avg %f tau %f" % (loss_receiver, reward.sum() / batch_size, sumrev.cpu().data.numpy() / (100*batch_size), tau))
            sumrev = 0

        if i % 5000 == 0:
            print_language(a, data)


if __name__ == "__main__":
    main()





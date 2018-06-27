import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
from torch.distributions import Normal

import torchHelpers as th

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
         env=None,
         agent_a=None,
         agent_b=None,
         com_model='onehot',
         com_noise=0,
         msg_dim=11,
         max_epochs=1000,
         noise_level=0,
         batch_size=100,
         sender_loss_multiplier=100,
         print_interval=1000,
         perception_dim=3,
         reward_func='regier_reward'):

    optimizer = optim.Adam(list(agent_a.parameters()) + list(agent_b.parameters()))  #optim.Adam(list(agent_a.parameters()))
    criterion_receiver = torch.nn.CrossEntropyLoss()

    sumrev = 0
    for i in range(max_epochs):
        optimizer.zero_grad()

        color_codes, colors = env.batch(batch_size=batch_size)
        color_codes = th.long_var(color_codes, cuda)
        colors = th.float_var(colors, cuda)

        noise = th.float_var(Normal(torch.zeros(batch_size, perception_dim),
                                    torch.ones(batch_size, perception_dim) * noise_level).sample(), cuda)
        colors = colors + noise

        probs = agent_a(perception=colors)

        if com_model == 'onehot':
            m = Categorical(probs)
            msg = m.sample()

            color_guess = agent_b(msg=msg)

            if reward_func == 'basic_reward':
                reward = env.basic_reward(color_codes, color_guess)
            elif reward_func == 'regier_reward':
                reward = env.regier_reward(colors, color_guess, cuda)

            sumrev += reward.sum()

            loss_sender = sender_loss_multiplier * ((-m.log_prob(msg) * reward).sum()/batch_size)

        elif com_model == 'softmax':
            loss_sender = 0
            noise = th.float_var(Normal(torch.zeros(batch_size, msg_dim),
                                        torch.ones(batch_size, msg_dim) * com_noise).sample(), cuda)
            msg=probs+noise
            color_guess = agent_b(msg=msg)


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


    return agent_a.cpu()


# Script entry point
if __name__ == "__main__":
    agent = main()


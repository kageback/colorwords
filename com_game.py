import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
from torch.distributions import Normal

import torchHelpers as th

def color_graph_V(a, env):
    V = {}
    a = th.cuda(a)
    chip_indices, colors = env.all_colors()
    colors = th.float_var(colors)

    probs = a(perception=colors)
    _, words = probs.max(1)

    for chip_index in chip_indices:
        V[chip_index] = words[chip_index].cpu().data[0]

    return V


# Model training loop

def play_noisy_channel_cont_reward(env,
                                   agent_a,
                                   agent_b,
                                   com_noise=0,
                                   msg_dim=11,
                                   max_epochs=1000,
                                   perception_noise=0,
                                   batch_size=100,
                                   print_interval=1000,
                                   perception_dim=3):

    agent_a = th.cuda(agent_a)
    agent_b = th.cuda(agent_b)


    optimizer = optim.Adam(list(agent_a.parameters()) + list(agent_b.parameters()))
    criterion_receiver = torch.nn.CrossEntropyLoss()

    for i in range(max_epochs):
        optimizer.zero_grad()

        color_codes, colors = env.batch(batch_size=batch_size)
        color_codes = th.long_var(color_codes)
        colors = th.float_var(colors)

        # add perceptual noise
        noise = th.float_var(Normal(torch.zeros(batch_size, perception_dim),
                                    torch.ones(batch_size, perception_dim) * perception_noise).sample())
        colors = colors + noise

        # generate message
        probs = agent_a(perception=colors)

        # add communication noise
        noise = th.float_var(Normal(torch.zeros(batch_size, msg_dim),
                                    torch.ones(batch_size, msg_dim) * com_noise).sample())
        msg=probs+noise

        # interpret message
        color_guess = agent_b(msg=msg)

        #compute loss and update model
        loss = criterion_receiver(color_guess, color_codes)
        loss.backward()
        optimizer.step()

        # printing status
        if print_interval != 0 and (i % print_interval == 0):
            print("Loss %f Naive perplexity %f" %
                  (loss,
                   torch.exp(loss))
                  )

    return agent_a.cpu()



def play_onehot(env,
                agent_a,
                agent_b,
                max_epochs=1000,
                noise_level=0,
                batch_size=100,
                sender_loss_multiplier=100,
                print_interval=1000,
                perception_dim=3,
                reward_func='regier_reward'):


    agent_a = th.cuda(agent_a)
    agent_b = th.cuda(agent_b)


    optimizer = optim.Adam(list(agent_a.parameters()) + list(agent_b.parameters()))
    criterion_receiver = torch.nn.CrossEntropyLoss()

    sumrev = 0
    for i in range(max_epochs):
        optimizer.zero_grad()

        color_codes, colors = env.batch(batch_size=batch_size)
        color_codes = th.long_var(color_codes)
        colors = th.float_var(colors)

        noise = th.float_var(Normal(torch.zeros(batch_size, perception_dim),
                                    torch.ones(batch_size, perception_dim) * noise_level).sample())
        colors = colors + noise

        probs = agent_a(perception=colors)


        m = Categorical(probs)
        msg = m.sample()

        color_guess = agent_b(msg=msg)

        if reward_func == 'basic_reward':
            reward = env.basic_reward(color_codes, color_guess)
        elif reward_func == 'regier_reward':
            reward = env.regier_reward(colors, color_guess)

        sumrev += reward.sum()

        loss_sender = sender_loss_multiplier * ((-m.log_prob(msg) * reward).sum()/batch_size)

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
    agent = play_noisy_channel_cont_reward()


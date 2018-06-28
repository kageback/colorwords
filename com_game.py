import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
from torch.distributions import Normal

import torchHelpers as th


# immutable game classes
class BaseGame:

    def __init__(self,
                 max_epochs=1000,
                 batch_size=100,
                 print_interval=1000):
        super().__init__()

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.print_interval = print_interval

    def play(self, env, agent_a, agent_b):

        agent_a = th.cuda(agent_a)
        agent_b = th.cuda(agent_b)

        optimizer = optim.Adam(list(agent_a.parameters()) + list(agent_b.parameters()))

        for i in range(self.max_epochs):
            optimizer.zero_grad()

            color_codes, colors = env.batch(batch_size=self.batch_size)
            color_codes = th.long_var(color_codes)
            colors = th.float_var(colors)

            loss = self.communication_channel(agent_a, agent_b, color_codes, colors)

            loss.backward()
            optimizer.step()

            # printing status
            if self.print_interval != 0 and (i % self.print_interval == 0):
                self.print_status(loss)

        return agent_a.cpu()

    def communication_channel(self, agent_a, agent_b, color_codes, colors):
        pass

    def print_status(self, loss):
        print("Loss %f Naive perplexity %f" %
              (loss,
               torch.exp(loss))
              )


class NoisyChannelContRewardGame(BaseGame):

    def __init__(self,
                 com_noise=0,
                 msg_dim=11,
                 max_epochs=1000,
                 perception_noise=0,
                 batch_size=100,
                 print_interval=1000,
                 perception_dim=3):
        super().__init__(max_epochs, batch_size, print_interval)

        self.com_noise = com_noise
        self.msg_dim = msg_dim
        self.perception_noise = perception_noise
        self.perception_dim = perception_dim

        self.criterion_receiver = torch.nn.CrossEntropyLoss()

    def communication_channel(self, agent_a, agent_b, color_codes, colors):
        # add perceptual noise
        noise = th.float_var(Normal(torch.zeros(self.batch_size, self.perception_dim),
                                    torch.ones(self.batch_size, self.perception_dim) * self.perception_noise).sample())
        colors = colors + noise
        # generate message
        probs = agent_a(perception=colors)
        # add communication noise
        noise = th.float_var(Normal(torch.zeros(self.batch_size, self.msg_dim),
                                    torch.ones(self.batch_size, self.msg_dim) * self.com_noise).sample())
        msg = probs + noise
        # interpret message
        color_guess = agent_b(msg=msg)
        # compute loss and update model
        loss = self.criterion_receiver(color_guess, color_codes)
        return loss




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


def play_noisy_channel_desc_reward(env,
                                   agent_a,
                                   agent_b,
                                   com_noise=0,
                                   msg_dim=11,
                                   max_epochs=1000,
                                   batch_size=100,
                                   print_interval=1000,
                                   perception_dim=3):
    pass


# Script entry point
if __name__ == "__main__":
    agent = play_noisy_channel_cont_reward()


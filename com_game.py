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

        self.training_mode = True

    def play(self, env, agent_a, agent_b):
        agent_a = th.cuda(agent_a)
        agent_b = th.cuda(agent_b)

        optimizer = optim.Adam(list(agent_a.parameters()) + list(agent_b.parameters()))

        for i in range(self.max_epochs):
            optimizer.zero_grad()

            color_codes, colors = env.mini_batch(batch_size=self.batch_size)
            color_codes = th.long_var(color_codes)
            colors = th.float_var(colors)

            loss = self.communication_channel(env, agent_a, agent_b, color_codes, colors)

            loss.backward()
            optimizer.step()

            # printing status
            if self.print_interval != 0 and (i % self.print_interval == 0):
                self.print_status(loss)

        return agent_a.cpu()

    def communication_channel(self, env, agent_a, agent_b, color_codes, colors):
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

        self.sum_reward = 0

    def communication_channel(self, env, agent_a, agent_b, target, perception):
        # add perceptual noise
        if self.training_mode:
            noise = th.float_var(Normal(torch.zeros(self.batch_size, self.perception_dim),
                                        torch.ones(self.batch_size, self.perception_dim) * self.perception_noise).sample())
            perception = perception + noise
        # generate message
        msg_probs = agent_a(perception=perception)
        # add communication noise
        if self.training_mode:
            noise = th.float_var(Normal(torch.zeros(self.batch_size, self.msg_dim),
                                        torch.ones(self.batch_size, self.msg_dim) * self.com_noise).sample())
            msg = msg_probs + noise
        # interpret message and sample a guess
        guess_probs = agent_b(msg=msg)
        m = Categorical(guess_probs)
        guess = m.sample()

        #compute reward

        CIELAB_guess = th.float_var(env.chip_index2CIELAB(guess.data))
        reward = env.sim(perception, CIELAB_guess)

        #reward = env.regier_reward(perception, guess_probs)

        self.sum_reward += reward.sum()
        # compute loss and update model
        loss = (-m.log_prob(guess) * reward).sum() / self.batch_size
        return loss

    def print_status(self, loss):

        print("Loss %f Naive perplexity %f Average reward %f" %
              (loss, torch.exp(loss), self.sum_reward / (self.print_interval * self.batch_size))
              )
        self.sum_reward = 0

class OneHotChannelContRewardGame(BaseGame):
    def __init__(self,
                 reward_func='regier_reward',
                 sender_loss_multiplier=100,
                 msg_dim=11,
                 max_epochs=1000,
                 perception_noise=0,
                 batch_size=100,
                 print_interval=1000,
                 perception_dim=3):
        super().__init__(max_epochs, batch_size, print_interval)
        self.reward_func = reward_func
        self.sender_loss_multiplier = sender_loss_multiplier
        self.msg_dim = msg_dim
        self.perception_noise = perception_noise
        self.perception_dim = perception_dim

        self.criterion_receiver = torch.nn.CrossEntropyLoss()
        self.sum_reward = 0

    def communication_channel(self, env, agent_a, agent_b, target, perception):
        # add perceptual noise
        if self.training_mode:
            noise = th.float_var(Normal(torch.zeros(self.batch_size, self.perception_dim),
                                        torch.ones(self.batch_size, self.perception_dim) * self.perception_noise).sample())
            perception = perception + noise
        # Sample message
        probs = agent_a(perception=perception)
        m = Categorical(probs)
        msg = m.sample()
        # interpret message
        guess = agent_b(msg=msg)
        # compute reward
        if self.reward_func == 'basic_reward':
            reward = env.basic_reward(target, guess)
        elif self.reward_func == 'regier_reward':
            reward = env.regier_reward(perception, guess)
        self.sum_reward += reward.sum()
        # compute loss
        self.loss_sender = self.sender_loss_multiplier * ((-m.log_prob(msg) * reward).sum() / self.batch_size)
        self.loss_receiver = self.criterion_receiver(guess, target)
        return self.loss_receiver + self.loss_sender

    def print_status(self, loss):

        print("Loss sender %f Loss receiver %f Naive perplexity %f Average reward %f" %
              (self.loss_sender,
               self.loss_receiver,
               torch.exp(self.loss_receiver), self.sum_reward / (self.print_interval * self.batch_size))
              )
        self.sum_reward = 0

# def play_onehot(env,
#                 agent_a,
#                 agent_b,
#                 max_epochs=1000,
#                 noise_level=0,
#                 batch_size=100,
#                 sender_loss_multiplier=100,
#                 print_interval=1000,
#                 perception_dim=3,
#                 reward_func='regier_reward'):
#
#
#     agent_a = th.cuda(agent_a)
#     agent_b = th.cuda(agent_b)
#
#
#     optimizer = optim.Adam(list(agent_a.parameters()) + list(agent_b.parameters()))
#     criterion_receiver = torch.nn.CrossEntropyLoss()
#
#     sumrev = 0
#     for i in range(max_epochs):
#         optimizer.zero_grad()
#
#         color_codes, colors = env.batch(batch_size=batch_size)
#         color_codes = th.long_var(color_codes)
#         colors = th.float_var(colors)
#
#         noise = th.float_var(Normal(torch.zeros(batch_size, perception_dim),
#                                     torch.ones(batch_size, perception_dim) * noise_level).sample())
#         colors = colors + noise
#
#         probs = agent_a(perception=colors)
#
#         m = Categorical(probs)
#         msg = m.sample()
#
#         color_guess = agent_b(msg=msg)
#
#         if reward_func == 'basic_reward':
#             reward = env.basic_reward(color_codes, color_guess)
#         elif reward_func == 'regier_reward':
#             reward = env.regier_reward(colors, color_guess)
#
#         sumrev += reward.sum()
#
#         loss_sender = sender_loss_multiplier * ((-m.log_prob(msg) * reward).sum() / batch_size)
#
#         loss_receiver = criterion_receiver(color_guess, color_codes)
#         loss = loss_receiver + loss_sender
#
#         loss.backward()
#
#         optimizer.step()
#
#         # printing status and periodic evaluation
#
#         if print_interval != 0 and (i % print_interval == 0):
#             print("Loss sender %f Loss receiver %f Naive perplexity %f Average reward %f" %
#                   (loss_sender,
#                    loss_receiver,
#                    torch.exp(loss_receiver), sumrev / (print_interval*batch_size))
#                   )
#             sumrev = 0
#
#
#     return agent_a.cpu()


# Script entry point
import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

class PopulationGame:
    def __init__(self,
                 max_epochs=10,
                 batch_size=10,
                 perception_dim=3,
                 perception_noise=0,
                 print_interval=1000,
                 reward_func='regier_reward',
                 evaluate_interval=0,
                 log_path=''):

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.perception_dim = perception_dim
        self.perception_noise = perception_noise
        self.reward_func=reward_func
        self.print_interval=print_interval
        self.evaluate_interval = evaluate_interval
        self.log_path = log_path

    def play(self, graph, env):
        parameters = []
        # For variance reduction
        sender_baselines = torch.zeros(graph.n_agents)
        listener_baselines = torch.zeros(graph.n_agents)
        sender_iterations = torch.zeros(graph.n_agents)
        listener_iterations = torch.zeros(graph.n_agents)

        for agent in graph.agents:
            parameters += list(agent.parameters())
        optimizer = optim.Adam(parameters)
        for t in range(self.max_epochs):
            pop_reward = 0
            optimizer.zero_grad()
            loss = 0
            # Generate messages for each agent
            for i in range(graph.n_agents):
                sender_loss = 0
                # Sample neighbors according to the graph
                conversations = graph.sample_neighbor(i, batch_size=self.batch_size)
                conversations.sort()
                listeners = list(set(conversations))
                # Keep track on which msg to which listener
                comm_mapping = {j : [True if x == j else False for _, x in enumerate(conversations)] for j in listeners}
                # Add noise
                targets, perception = env.mini_batch(batch_size=self.batch_size)
                perception = torch.FloatTensor(perception)
                noise = Normal(torch.zeros(self.batch_size, self.perception_dim),
                               torch.ones(self.batch_size, self.perception_dim) * self.perception_noise).sample()
                perception = perception + noise
                # Generate messages
                msg_logits = graph.agents[i](perception=perception)
                msg_probs = F.softmax(msg_logits, dim=1)
                msg_dist = Categorical(msg_probs)
                msg = msg_dist.sample()
                # Distribute messages
                tmp_reward = 0
                for j in listeners:
                    # Get all msgs to agent j
                    ind = comm_mapping[j]
                    guess_logits = graph.agents[j](msg=msg[ind])
                    guess_probs = F.softmax(guess_logits, dim=1)
                    guess_dist = Categorical(guess_probs)
                    guess = guess_dist.sample()
                    if self.reward_func == 'regier_reward':
                        CIELAB_guess = env.chip_index2CIELAB(guess.data)
                        CIELAB_target = env.chip_index2CIELAB(targets)
                        reward = env.regier_reward(perception[ind], CIELAB_guess)

                    # Compute losses
                    sender_loss += - (msg_dist.log_prob(msg)[ind] * (reward - sender_baselines[i])).sum()
                    listener_loss = -(guess_dist.log_prob(guess) * (reward - listener_baselines[j]))
                    loss += listener_loss.mean()
                    # Update baselines
                    listener_iterations[j] += 1
                    listener_baselines[j] += (reward.mean() - listener_baselines[j]) / listener_iterations[j]

                    tmp_reward += reward.detach().cpu().sum()
                sender_iterations[i] += 1
                sender_baselines[i] += (tmp_reward.sum() / self.batch_size - sender_baselines[i]) / sender_iterations[i]
                pop_reward += tmp_reward.numpy()/self.batch_size
                # We would like to take mean over all conversations
                loss += sender_loss.mean()

            loss.backward()
            optimizer.step()

            pop_reward = pop_reward / graph.n_agents

            if t != 0 and t % self.print_interval == 0:
                print('Average reward/conversation : {0}, Total loss : {1}'.format(pop_reward, loss))

        return deepcopy(graph.agents[0])







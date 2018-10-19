import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
from torch.distributions import Normal

import evaluate
import torchHelpers as th
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# immutable(ish) game classes. Not meant to carry state between executions since each execution is based on the object
# created in the run script and not updated with new state after running on the cluster.
class BaseGame:

    def __init__(self,
                 max_epochs=1000,
                 batch_size=100,
                 print_interval=1000,
		         evaluate_interval=0,
                 log_path=''):
        super().__init__()

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.print_interval = print_interval
        self.evaluate_interval = evaluate_interval
        self.log_path = log_path
        self.training_mode = True

        self.gibson_cost = []
        self.regier_cost = []
        self.wellformedness = []
        self.term_usage = []

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
            if self.print_interval != 0 and ((i+1) % self.print_interval == 0):
                self.print_status(loss)

            if self.evaluate_interval != 0 and ((i+1) % self.evaluate_interval == 0):
                self.evaluate(env, agent_a)

        return agent_a.cpu()

    def communication_channel(self, env, agent_a, agent_b, color_codes, colors):
        pass

    def evaluate(self, env, agent_a):
        V = evaluate.agent_language_map(env, agent_a)
        self.gibson_cost += [evaluate.compute_gibson_cost2(env, a=agent_a)[1]]
        self.regier_cost += [evaluate.communication_cost_regier(env, V=V)[0]]
        self.wellformedness += [evaluate.wellformedness(env, V=V)[0]]
        self.term_usage += [evaluate.compute_term_usage(V=V)]
        print('terms = {:2d}, gib = {:.3f}, reg = {:.3f}, well = {:.3f}'.format(self.term_usage[-1],
                                                                                self.gibson_cost[-1],
                                                                                self.regier_cost[-1],
                                                                                self.wellformedness[-1]))
        env.plot_with_colors(V, save_to_path='{}evo_map-{}_terms.png'.format(self.log_path, self.term_usage[-1]))

        plt.figure()
        plt.plot(self.gibson_cost)
        plt.savefig('{}gibson_cost_evo.png'.format(self.log_path))
        plt.figure()
        plt.plot(self.regier_cost)
        plt.savefig('{}regier_cost_evo.png'.format(self.log_path))
        plt.figure()
        plt.plot(self.wellformedness)
        plt.savefig('{}wellformedness_evo.png'.format(self.log_path))
        plt.figure()
        plt.plot(self.term_usage)
        plt.savefig('{}term_usage_evo.png'.format(self.log_path))

    # other metrics
    # Outdated as of nosy channel model
    def compute_gibson_cost(self, env, a):
        _, perceptions = env.full_batch()
        perceptions = perceptions.cpu()
        all_terms = th.long_var(range(a.msg_dim), False)

        p_WC = F.softmax(a(perception=perceptions), dim=1).t().data.numpy()

        p_CW = F.softmax(a(msg=all_terms), dim=1).data.numpy()

        S = -np.diag(np.matmul(p_WC.transpose(), (np.log2(p_CW))))
        avg_S = S.sum() / len(S)  # expectation assuming uniform prior
        # debug code
        # s = 0
        # c = 43
        # for w in range(a.msg_dim):
        #     s += -p_WC[w, c]*np.log2(p_CW[w, c])
        # print(S[c] - s)
        return S, avg_S


    @staticmethod
    def reduce_maps(name, exp, reduce_method='mode'):
        maps = exp.get_flattened_results(name)
        np_maps = np.array([list(map.values()) for map in maps])
        if reduce_method == 'mode':
            np_mode_map = stats.mode(np_maps).mode[0]
            res = {k: np_mode_map[k] for k in maps[0].keys()}
        else:
            raise ValueError('unsupported reduce function: ' + reduce_method)
        return res

    @staticmethod
    def compute_ranges(V):
        lex = {}
        for n in range(len(V)):
            if not V[n] in lex.keys():
                lex[V[n]] = [n]
            else:
                lex[V[n]] += [n]
        ranges = {}
        for w in lex.keys():
            ranges[w] = []
            state = 'out'
            for n in lex[w]:
                if state == 'out':
                    range_start = n
                    prev = n
                    state = 'in'
                elif state == 'in':
                    if prev + 1 != n:
                        ranges[w] += [(range_start, prev)]
                        range_start = n
                    prev = n
            ranges[w] += [(range_start, prev)]
        return ranges

    def print_status(self, loss):
        print("Loss %f Naive perplexity %f" %
              (loss,
               torch.exp(loss))
              )

class NoisyChannelGame(BaseGame):

    def __init__(self,
                 reward_func='regier_reward',
                 bw_boost=0,
                 com_noise=0,
                 msg_dim=11,
                 max_epochs=1000,
                 perception_noise=0,
                 batch_size=100,
                 print_interval=1000,
                 evaluate_interval=0,
                 log_path='',
                 perception_dim=3,
                 loss_type='CrossEntropyLoss'):
        super().__init__(max_epochs, batch_size, print_interval, evaluate_interval, log_path)
        self.reward_func = reward_func
        self.bw_boost = bw_boost
        self.com_noise = com_noise
        self.msg_dim = msg_dim
        self.perception_noise = perception_noise
        self.perception_dim = perception_dim

        self.loss_type = loss_type

        self.sum_reward = 0

        self.criterion_receiver = torch.nn.CrossEntropyLoss()

    def communication_channel(self, env, agent_a, agent_b, target, perception):
        # add perceptual noise

        noise = th.float_var(Normal(torch.zeros(self.batch_size, self.perception_dim),
                                    torch.ones(self.batch_size, self.perception_dim) * self.perception_noise).sample())
        perception = perception + noise
        # generate message
        msg_logits = agent_a(perception=perception)
        # add communication noise
        # msg_probs = F.gumbel_softmax(msg_logits, tau=2 / 3)
        noise = th.float_var(Normal(torch.zeros(self.batch_size, self.msg_dim),
                                    torch.ones(self.batch_size, self.msg_dim) * self.com_noise).sample())
        msg_probs = F.softmax(msg_logits + noise, dim=1)

        # interpret message and sample a guess
        guess_logits = agent_b(msg=msg_probs)
        guess_probs = F.softmax(guess_logits, dim=1)
        m = Categorical(guess_probs)
        guess = m.sample()

        #compute reward
        if self.reward_func == 'regier_reward':
            CIELAB_guess = env.chip_index2CIELAB(guess.data)
            reward = env.regier_reward(perception, CIELAB_guess, bw_boost=self.bw_boost)
        elif self.reward_func == 'abs_dist':
            diff = torch.abs(target - guess.unsqueeze(dim=1))
            reward = 1-(diff.float()/100) #1-(diff.float()/50)

        self.sum_reward += reward.sum()

        # compute loss and update model
        if self.loss_type =='REINFORCE':
            loss = (-m.log_prob(guess) * reward).sum() / self.batch_size
        elif self.loss_type == 'CrossEntropyLoss':
            loss = self.criterion_receiver(guess_logits, target.squeeze())

        return loss

    def print_status(self, loss):

        print("Loss %f Average reward %f" %
              (loss, self.sum_reward / (self.print_interval * self.batch_size))
              )
        self.sum_reward = 0

class OneHotChannelGame(BaseGame):
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

import torch
from torch import optim
from torch.autograd import Variable
from torch.distributions import Categorical

import data
import agents


print(data.data[:5])
#tmp_msg = Variable(torch.LongTensor([0,1]))
tmp_color_code = Variable(torch.LongTensor([0, 1]))
tmp_color = Variable(torch.FloatTensor([[0,0,0],[1,1,1]]))


msg_dim = 2
hidden_dim = 2
color_dim = 2
perception_dim = 3


a = agents.BasicAgent(msg_dim, hidden_dim, color_dim, perception_dim)

optimizer = optim.Adam(a.parameters())

criterion_receiver = torch.nn.CrossEntropyLoss()

for i in range(100000):
    optimizer.zero_grad()

    probs = a(perception=tmp_color)
    m = Categorical(probs)
    msg = m.sample()

    color_guess = a(msg=msg)

    loss_receiver = criterion_receiver(color_guess, tmp_color_code)

    _, I = color_guess.max(1)
    reward = (tmp_color_code == I).float() - (tmp_color_code != I).float()

    loss_sender = -m.log_prob(msg) * reward
    loss_sender = loss_sender.sum()

    loss = loss_receiver + loss_sender
    loss.backward()

    optimizer.step()

    if i % 100 == 0:
        print("error %f reward %d" % (loss_receiver, reward.sum()) )

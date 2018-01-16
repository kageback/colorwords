import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicAgent(nn.Module):

    def __init__(self, msg_dim, hidden_dim, color_dim, perception_dim):
        super().__init__()

        # Receiving part
        self.msg_receiver = nn.Embedding(msg_dim, hidden_dim)
        self.color_estimator = nn.Linear(hidden_dim,color_dim)

        #Sending part
        self.perception_embedding = nn.Linear(perception_dim,hidden_dim)
        self.msg_creator = nn.Linear(hidden_dim,msg_dim)


    def forward(self, perception=None, msg=None, tau=1):

        if msg is not None:
            h = F.tanh(self.msg_receiver(msg))
            color_logits = self.color_estimator(h)
            return color_logits

        if perception is not None:
            h = F.tanh(self.perception_embedding(perception))

            probs = F.softmax(self.msg_creator(h)/tau, dim=1)
            return probs




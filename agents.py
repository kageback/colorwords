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
        self.msg_creator = nn.Linear(perception_dim,msg_dim)


    def forward(self, perception=None, msg=None):

        if not msg is None:
            h = self.msg_receiver(msg)
            color_logits = self.color_estimator(h)
            return color_logits

        if not perception is None:
            msg_logits = self.msg_creator(perception)
            probs = F.softmax(msg_logits, dim=1
)
            return probs
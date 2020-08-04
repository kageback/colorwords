import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

class BaseGame:

    def __init__(self,
                 max_epochs=1000,
                 batch_size=100,
                 print_interval=1000,
                 evaluate_interval=0,
                 agents=None,
                 graph=None,
                 log_path=''):
        super().__init__()


    def get_population(self):
        pass

    def set_population(self):
        pass

    def compute_graph(self):
        pass
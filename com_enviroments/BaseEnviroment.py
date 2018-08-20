import numpy as np
import torchHelpers as th
import torch.nn.functional as F

class BaseEnviroment:
    def __init__(self) -> None:
        super().__init__()

    def full_batch(self):
        pass

    def mini_batch(self, batch_size=10):
        pass

    def sim_index(self):
        pass




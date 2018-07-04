import numpy as np

from com_enviroments.BaseEnviroment import BaseEnviroment

class NumberEnvironment(BaseEnviroment):
    def __init__(self) -> None:
        super().__init__()
        self.data_dim = 100
        self.numbers = np.array(list(range(self.data_dim)))


    def full_batch(self):
        return self.numbers, np.expand_dims(self.numbers, axis=1)

    def mini_batch(self, batch_size=10):
        batch = np.expand_dims(np.random.randint(0, self.data_dim, batch_size), axis=1)
        return batch, batch

    def sim_np(self, num_a, num_b):
        return self.data_dim - np.sqrt(np.power(num_a-num_b, 2))

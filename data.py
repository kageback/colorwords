import pandas as pd
import numpy as np

data = pd.read_csv('data/cnum-vhcm-lab-new.txt', sep='\t')
data = data.apply(np.random.permutation)




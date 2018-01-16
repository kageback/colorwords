import pandas as pd
import numpy as np


def print_cnum(t):
    return str(t['#cnum'].values[0])

class WCSColorData:


    def __init__(self, wcs_path = 'data/', limit_colors=0):
        self.data = pd.read_csv(wcs_path + 'cnum-vhcm-lab-new.txt', sep='\t')

        if limit_colors > 0:
            # make problem easier by limiting the number of color cards used
            sample = self.data.sample(n=limit_colors)
            self.data = sample.reset_index()

        self.color_dim = len(self.data)


    def batch(self, batch_size = 10):
        batch = self.data.sample(n=batch_size, replace=True)

        return np.array(batch['#cnum'].index), batch[['L*','a*','b*']].values


    def all_colors(self):
        return np.array(self.data.index), np.array(self.data['#cnum']), self.data[['L*','a*','b*']].values


    def print(self, f=print_cnum, pad=3):
        # print x axsis
        print(''.ljust(pad), end="")
        for x in range(41):
            print(str(x).ljust(pad), end="")
        print('')

        # print color codes
        for y in list('ABCDEFGHIJ'):
            print(y.ljust(pad), end="")
            for x in range(41):
                t = self.data.loc[(self.data['H'] == x) & (self.data['V'] == y)]
                if len(t) == 0:
                    s = ''
                elif len(t) == 1:
                    s = f(t)
                else:
                    raise TabError()

                print(s.ljust(pad), end="")
            print('')







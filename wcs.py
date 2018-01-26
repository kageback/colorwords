import pandas as pd
import numpy as np




wcs_path = 'data/'
color_chips = pd.read_csv(wcs_path + 'cnum-vhcm-lab-new.txt', sep='\t')


def all_colors():
    return color_chips.index.values,color_chips['#cnum'].values, color_chips[['L*', 'a*', 'b*']].values  #


def batch(batch_size = 10):
    batch = color_chips.sample(n=batch_size, replace=True)

    return batch.index.values, batch[['L*', 'a*', 'b*']].values


def color_dim():
    return len(color_chips)


def chip_index2CIELAB(color_codes):
    rows = color_chips.loc[color_codes.data]

    return rows[['L*', 'a*', 'b*']].values

# Printing

def print_cnum(t):
    return str(t['#cnum'].values[0])


def print_color_map(f=print_cnum, pad=3):
    # print x axsis
    print(''.ljust(pad), end="")
    for x in range(41):
        print(str(x).ljust(pad), end="")
    print('')

    # print color codes
    for y in list('ABCDEFGHIJ'):
        print(y.ljust(pad), end="")
        for x in range(41):
            t = color_chips.loc[(color_chips['H'] == x) & (color_chips['V'] == y)]
            if len(t) == 0:
                s = ''
            elif len(t) == 1:
                s = f(t)
            else:
                raise TabError()

            print(s.ljust(pad), end="")
        print('')







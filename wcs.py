import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as lines


wcs_path = 'data/'
color_chips = pd.read_csv(wcs_path + 'cnum-vhcm-lab-new.txt', sep='\t')
cielab_map = color_chips[['L*', 'a*', 'b*']].values


def all_colors():
    return color_chips.index.values, cielab_map


def batch(batch_size = 10):
    batch = color_chips.sample(n=batch_size, replace=True)

    return batch.index.values, batch[['L*', 'a*', 'b*']].values


def color_dim():
    return len(color_chips)


def chip_index2CIELAB(color_codes):
    return cielab_map[color_codes]


# Evaluation metrics

def communication_cost_regier(V, sum_over_whole_s=False, norm_over_s=False, weight_by_size=False):

    s = {}
    for i in V.keys():
        s[i] = 0
        for j in V.keys():
            if V[i]['word'] == V[j]['word']:
                s[i] += sim(i, j)


    l = {}
    for t in V.keys():
        z = 0
        cat_size = 0
        for i in V.keys():

            if sum_over_whole_s or V[i]['word'] == V[t]['word']:
                z += s[i]
                cat_size += 1
        l[t] = s[t]/z
        if weight_by_size:
            l[t] *= cat_size/len(V)

    if norm_over_s:
        l_z=0
        for x in l.values():
            l_z += x
        for i in l.keys():
            l[i] /= l_z

    # debug code to check it l sums to one
    l_z=0
    for x in l.values():
        l_z += x

    E = 0
    for t in V.keys():
        E += -np.log2(l[t])
    E = E / len(V)

    return E


def wellformedness(V):
    Sw = 0
    for i in V.keys():
        for j in V.keys():
            if V[i]['word'] == V[j]['word']:
                Sw += sim(i, j)
    Da = 0
    for i in V.keys():
        for j in V.keys():
            if V[i]['word'] != V[j]['word']:
                Da += 1- sim(i, j)
    W = Sw + Da
    return W


def combined_criterion(V):
    def inc_dict(dict, key, increment):
        if key in dict.keys():
            dict[key] += increment
        else:
            dict[key] = increment


    cat_sizes = {}
    for v in V.values():
        inc_dict(cat_sizes,v['word'],1)

    n = len(cat_sizes)

    cost = 0
    for w in cat_sizes.keys():
        cost += cat_sizes[w]*np.log2(cat_sizes[w])/n

    ## CCP part
    CCP = 0
    for i in V.keys():
        for j in V.keys():
            if V[i]['word'] != V[j]['word']:
                CCP += 2*sim(i, j)-1

    return cost + CCP

def min_k_cut_cost(V, k):
    def xrange(start, stop):
        return range(start, stop + 1)

    C = {}
    for i in xrange(1, k):
        C[i] = []

    for chip_index in V.keys():
        C[V[chip_index]['word']+1].append(chip_index)

    cost = 0
    for i in xrange(1, k-1):
        for j in xrange(i+1, k):
            for v1 in C[i]:
                for v2 in C[j]:
                    cost += sim(v1, v2)
    return cost


def sim(chip_index_x, chip_index_y, c=0.001):

    color_x = cielab_map[chip_index_x]
    color_y = cielab_map[chip_index_y]

    # CIELAB distance 76 (euclidean distance)
    d = np.linalg.norm(color_x - color_y, 2)

    # Regier color similarity
    return np.exp(-c * np.power(d, 2))


# Printing

def print_cnum(t):
    return str(t['#cnum'].values[0])

def print_index(t):
    return str(t.index.values[0])


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

# plotting
def cielab2rgb(c):
    from colormath.color_objects import LabColor, sRGBColor
    from colormath.color_conversions import convert_color

    lab = LabColor(c[0],c[1],c[2])
    rgb = convert_color(lab, sRGBColor)

    return np.array(rgb.get_value_tuple())


def plot_with_colors(V, save_to_path='dev.png', y_wcs_range=' ABCDEFGHIJ ', x_wcs_range=range(1, 41), use_real_color=True, add_boarders_color=''):
    #print_color_map(print_index, 4)

    N_x = len(x_wcs_range)
    N_y = len(y_wcs_range)
    # make an empty data set
    word = np.ones([N_y, N_x])
    rgb = np.ones([N_y, N_x, 3])
    for y_alpha, y in zip(list(y_wcs_range), range(N_y)):
        for x_wcs, x in zip(x_wcs_range, range(N_x)):
            t = color_chips.loc[(color_chips['H'] == x_wcs) & (color_chips['V'] == y_alpha)]
            if len(t) == 0:
                word[y, x] = np.nan
                rgb[y, x, :] = np.array([1, 1, 1])
            elif len(t) == 1:
                word[y, x] = V[t.index.values[0]]['word']
                rgb[y, x, :] = cielab2rgb(t[['L*', 'a*', 'b*']].values[0])
            else:
                raise TabError()

    fig, ax = plt.subplots(1, 1, tight_layout=True)

    if add_boarders_color != '':
        for y in range(N_y):
            for x in range(N_x):
                if not np.isnan(word[y, x]):
                    if x+1 < N_x and word[y, x] != word[y, x+1]:
                        ax.add_line(lines.Line2D([x+1, x+1], [N_y - y, N_y - (y+1)], color=add_boarders_color))

                    if (y+1 < N_y and word[y, x] != word[y+1, x]) or y+1 == N_y:
                            ax.add_line(lines.Line2D([x, x + 1], [N_y - (y + 1), N_y - (y + 1)], color=add_boarders_color))

                    if (y-1 >= 0 and word[y, x] != word[y-1, x]) or y-1 < 0:
                            ax.add_line(lines.Line2D([x, x + 1], [N_y - (y + 0), N_y - (y + 0)], color=add_boarders_color))


    #my_cmap = matplotlib.colors. ListedColormap(['r', 'g', 'b'])
    my_cmap = plt.get_cmap('Set1')
    my_cmap.set_bad(color='w', alpha=0)

    data = rgb if use_real_color else word
    ax.imshow(data, interpolation='none', cmap=my_cmap, extent=[0, N_x, 0, N_y], zorder=0)

    ax.axis('off')

    plt.savefig(save_to_path)
    plt.close()

    #1*10 + 40*8

    # Example code
    # fill in some fake data
    #for j in range(3)[::-1]:
    #    data[N // 2 - j: N // 2 + j + 1, N // 2 - j: N // 2 + j + 1] = j

    # make a figure + axes
    #fig, ax = plt.subplots(1, 1, tight_layout=True)

    # make color map
    #my_cmap = matplotlib.colors.ListedColormap(['r', 'g', 'b'])

    # set the 'bad' values (nan) to be white and transparent
    #my_cmap.set_bad(color='w', alpha=0)

    # draw the grid
    #for x in range(N + 1):
    #    ax.axhline(x, lw=2, color='k', zorder=5)
    #    ax.axvline(x, lw=2, color='k', zorder=5)

    # draw the boxes
    #ax.imshow(data, interpolation='none', cmap=my_cmap, extent=[0, N_x, 0, N_y], zorder=0)
    # turn off the axis labels
    #ax.axis('off')

    #plt.savefig('test.png')







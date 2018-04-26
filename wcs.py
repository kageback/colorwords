import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as lines


wcs_path = 'data/'
color_chips = pd.read_csv(wcs_path + 'cnum-vhcm-lab-new.txt', sep='\t')
cielab_map = color_chips[['L*', 'a*', 'b*']].values

term = pd.read_csv(wcs_path + 'term.txt', sep='\t', names=['lang_num', 'spkr_num', 'chip_num', 'term_abrev'])
dict = pd.read_csv(wcs_path + 'dict.txt', sep='\t', skiprows=[0], names=['lang_num', 'term_num', 'term', 'term_abrev'])
term_nums = pd.merge(term,
                     dict.drop_duplicates(subset=['lang_num', 'term_abrev']),
                     how='inner',
                     on=['lang_num', 'term_abrev'])

def language_map(lang_num):
    l = term_nums.loc[term_nums.lang_num == lang_num]
    map = {chip_i: l.loc[l.chip_num == chip_i]['term_num'].mode().values[0] for chip_i in range(1, 331)}
    return map

#Iduna (lang_num47)
#map = language_map(47)

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
import itertools
def compareMaps(A, B):
    wordset_A = {a['word'] for a in A.values()}
    wordset_B = {b['word'] for b in B.values()}
    if not len(wordset_A) == len(wordset_B):
        return 0
    if len(wordset_A) < 11:
        return

    max_overlap = 0
    for beta in itertools.permutations(wordset_B):
        overlap = compareAssignment(A, B, wordset_A, beta)
        if overlap > max_overlap:
            max_overlap = overlap
            best_beta = beta

    return max_overlap


def compareAssignment(A, B, word_order_A, word_order_B):
    scores = []
    for word_index_a, word_index_b in zip(word_order_A, word_order_B):
        a_tiles = {t[0] for t in A.items() if t[1]['word'] == word_index_a}
        b_tiles = {t[0] for t in B.items() if t[1]['word'] == word_index_b}
        scores.append(len(b_tiles.intersection(a_tiles)) / max([len(a_tiles), len(b_tiles)]))

    return min(scores)


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


def compute_term_usage(V):
    def inc_dict(dict, key, increment):
        if key in dict.keys():
            dict[key] += increment
        else:
            dict[key] = increment

    cat_sizes = {}
    for v in V.values():
        inc_dict(cat_sizes, v['word'], 1)
    n = len(cat_sizes)
    return n, cat_sizes


def combined_criterion(V):
    n, cat_sizes = compute_term_usage(V)

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


def plot_with_colors(V, save_to_path='dev.png', y_wcs_range='ABCDEFGHIJ', x_wcs_range=range(0, 41), use_real_color=True):
    #print_color_map(print_index, 4)

    N_x = len(x_wcs_range)
    N_y = len(y_wcs_range)
    # make an empty data set
    word = np.ones([N_y, N_x],dtype=np.int64) * -1
    rgb = np.ones([N_y, N_x, 3])
    for y_alpha, y in zip(list(y_wcs_range), range(N_y)):
        for x_wcs, x in zip(x_wcs_range, range(N_x)):
            t = color_chips.loc[(color_chips['H'] == x_wcs) & (color_chips['V'] == y_alpha)]
            if len(t) == 0:
                word[y, x] = -1
                rgb[y, x, :] = np.array([1, 1, 1])
            elif len(t) == 1:
                word[y, x] = V[t.index.values[0]]['word']
                rgb[y, x, :] = cielab2rgb(t[['L*', 'a*', 'b*']].values[0])
            else:
                raise TabError()

    fig, ax = plt.subplots(1, 1, tight_layout=True)


    my_cmap = plt.get_cmap('tab20')

    bo = 0.2
    lw = 1.5
    for y in range(N_y):
        for x in range(N_x):
            if word[y, x] >= 0:
                word_color = my_cmap.colors[word[y, x]]
                word_border = '-'
                if word[y, x] != word[y, (x+1) % N_x]:
                    ax.add_line(lines.Line2D([x + 1, x + 1], [N_y - y, N_y - (y + 1)], color='w'))
                    ax.add_line(lines.Line2D([x+1-bo, x+1-bo], [N_y - (y+bo), N_y - (y+1-bo)], color=word_color, ls=word_border, lw=lw))

                if word[y, x] != word[y, x-1 if x-1 >= 0 else N_x-1]:
                    ax.add_line(lines.Line2D([x+bo, x+bo], [N_y - (y+bo), N_y - (y+1-bo)], color=word_color, ls=word_border, lw=lw))


                if (y+1 < N_y and word[y, x] != word[y+1, x]) or y+1 == N_y:
                    ax.add_line(lines.Line2D([x+bo, x + 1-bo], [N_y - (y + 1-bo), N_y - (y + 1-bo)], color=word_color, ls=word_border, lw=lw))
                    ax.add_line(lines.Line2D([x, x + 1], [N_y - (y + 1), N_y - (y + 1)], color='w'))

                if (y-1 >= 0 and word[y, x] != word[y-1, x]) or y-1 < 0:
                        ax.add_line(lines.Line2D([x+bo, x + 1-bo], [N_y - (y + bo), N_y - (y + bo)], color=word_color, ls=word_border, lw=lw))


    #my_cmap = matplotlib.colors. ListedColormap(['r', 'g', 'b'])

    my_cmap.set_bad(color='w', alpha=0)
    data = rgb if use_real_color else word
    data = data.astype(np.float)
    data[data == -1] = np.nan
    ax.imshow(data, interpolation='none', cmap=my_cmap, extent=[0, N_x, 0, N_y], zorder=0)

    #ax.axis('off')
    ylabels = list(y_wcs_range)
    ylabels.reverse()
    plt.yticks([i+0.5 for i in range(len(y_wcs_range))], ylabels, fontsize=8)
    plt.xticks([i+0.5 for i in range(len(x_wcs_range))], x_wcs_range, fontsize=8)

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







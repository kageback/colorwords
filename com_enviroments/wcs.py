import pandas as pd
import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as lines


import torch
import torchHelpers as th
from com_enviroments.BaseEnviroment import BaseEnviroment

import urllib.request as request
import os

class WCS_Enviroment(BaseEnviroment):
    def __init__(self, wcs_path='data/') -> None:
        super().__init__()

        baseurl = 'http://www1.icsi.berkeley.edu/wcs/data/'
        self.get_data(baseurl + 'cnum-maps/cnum-vhcm-lab-new.txt', 'data/cnum-vhcm-lab-new.txt')
        self.get_data(baseurl + '20021219/txt/term.txt', 'data/term.txt')
        self.get_data(baseurl + '20041016/txt/dict.txt', 'data/dict.txt')

        # http://www1.icsi.berkeley.edu/wcs/data/cnum-maps/cnum-vhcm-lab-new.txt
        # http://www1.icsi.berkeley.edu/wcs/data/20021219/txt/term.txt
        # http://www1.icsi.berkeley.edu/wcs/data/20041016/txt/dict.txt

        self.color_chips = pd.read_csv(wcs_path + 'cnum-vhcm-lab-new.txt', sep='\t')
        self.cielab_map = self.color_chips[['L*', 'a*', 'b*']].values

        self.term = pd.read_csv(wcs_path + 'term.txt', sep='\t', names=['lang_num', 'spkr_num', 'chip_num', 'term_abrev'])
        self.dict = pd.read_csv(wcs_path + 'dict.txt', sep='\t', skiprows=[0], names=['lang_num', 'term_num', 'term', 'term_abrev'])
        self.term_nums = pd.merge(self.term,
                                  self.dict.drop_duplicates(subset=['lang_num', 'term_abrev']),
                                  how='inner',
                                  on=['lang_num', 'term_abrev'])

        self.human_mode_maps = self.compute_human_mode_maps(wcs_path)

    def compute_human_mode_maps(self, wcs_path):
        # Human mode maps
        fname = wcs_path + 'mode_maps.plk'
        if os.path.exists(fname):
            return pickle.load(open(fname, "rb"))

        print('Computing human mode maps. This should only happen the first run but will take a minute to two.')
        human_mode_maps = {}
        if not os.path.exists(wcs_path + 'mode_maps'):
            os.mkdir(wcs_path + 'mode_maps')
        for lang_num in range(1, 111):
            l = self.term_nums.loc[self.term_nums.lang_num == lang_num]
            if np.unique(l.chip_num.values).shape[0] == 330:
                print('processing lang {}'.format(lang_num))
                m = [l.loc[l.chip_num == self.color_chips.loc[chip_i]['#cnum']]['term_num'].mode().values[0] for
                     chip_i in range(330)]
                human_mode_maps[lang_num] = np.array(m)
                self.plot_with_colors(np.array(m), wcs_path + 'mode_maps/human_lang_' + str(lang_num) + '.png')
            else:
                print(
                    'Skipping lang {} due to incomplete data (experiment only consisting of {}/330 chips).'.format(
                        lang_num,
                        np.unique(l.chip_num.values).shape[0]))
        pickle.dump(human_mode_maps, open(fname, "wb"))
        print('Done computing mode maps')
        return human_mode_maps


    def get_data(self, url, local_name):
        if not os.path.exists(local_name):
            print('Downloading ' + url)
            print('saved as ' + local_name)
            request.urlretrieve(url, local_name)

    # Data
    def full_batch(self):
        return self.color_chips.index.values, self.cielab_map

    def mini_batch(self, batch_size = 10):
        batch = self.color_chips.sample(n=batch_size, replace=True)
        return batch.index.values, batch[['L*', 'a*', 'b*']].values

    # Properties

    def data_dim(self):
        return len(self.color_chips)

    def chip_index2CIELAB(self, color_codes):
        return self.cielab_map[color_codes]

    # Reward functions
    def basic_reward(self, color_codes, color_guess):
        _, I = color_guess.max(1)
        reward = (color_codes == I).float() - (color_codes != I).float()
        return reward

    def regier_reward(self, cielab_color_x, cielab_color_y, c=0.001, bw_boost=1):
        # CIELAB distance 76 (euclidean distance)
        diff = (cielab_color_x - cielab_color_y)
        if bw_boost != 1:
            bw_booster = torch.FloatTensor([bw_boost, 1, 1])
            bw_booster /= bw_booster.norm()
            diff = diff*bw_booster
        dist = diff.norm(2, 1)
        # Regier similarity
        return torch.exp(-c * torch.pow(dist, 2))

    def sim_index(self, chip_index_x, chip_index_y, bw_boost=1):
        # sim func used for computing the evaluation metrics
        color_x = th.float_var(self.cielab_map[chip_index_x]).unsqueeze(0)
        color_y = th.float_var(self.cielab_map[chip_index_y]).unsqueeze(0)
        return self.regier_reward(color_x, color_y)

    # plotting
    def plot_with_colors(self, V, save_to_path='dev.png', y_wcs_range='ABCDEFGHIJ', x_wcs_range=range(0, 41), use_real_color=True):
        #self.print_color_map(f=lambda t: str(t.index.values[0]), pad=4)

        N_x = len(x_wcs_range)
        N_y = len(y_wcs_range)
        # make an empty data set
        word = np.ones([N_y, N_x],dtype=np.int64) * -1
        rgb = np.ones([N_y, N_x, 3])
        for y_alpha, y in zip(list(y_wcs_range), range(N_y)):
            for x_wcs, x in zip(x_wcs_range, range(N_x)):
                t = self.color_chips.loc[(self.color_chips['H'] == x_wcs) & (self.color_chips['V'] == y_alpha)]
                if len(t) == 0:
                    word[y, x] = -1
                    rgb[y, x, :] = np.array([1, 1, 1])
                elif len(t) == 1:
                    word[y, x] = V[t.index.values[0]]
                    rgb[y, x, :] = self.cielab2rgb(t[['L*', 'a*', 'b*']].values[0])
                else:
                    raise TabError()

        fig, ax = plt.subplots(1, 1, tight_layout=True)
        my_cmap = plt.get_cmap('tab20')
        bo = 0.2
        lw = 1.5
        for y in range(N_y):
            for x in range(N_x):
                if word[y, x] >= 0:
                    word_color = my_cmap.colors[word[y, x] % 20]
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

    def cielab2rgb(self, c):
        from colormath.color_objects import LabColor, sRGBColor
        from colormath.color_conversions import convert_color
        lab = LabColor(c[0], c[1], c[2])
        rgb = convert_color(lab, sRGBColor)
        return np.array(rgb.get_value_tuple())

    # Printing
    def print_color_map(self, f=lambda t: str(t['#cnum'].values[0]), pad=3):
        # print x axsis
        print(''.ljust(pad), end="")
        for x in range(41):
            print(str(x).ljust(pad), end="")
        print('')

        # print color codes
        for y in list('ABCDEFGHIJ'):
            print(y.ljust(pad), end="")
            for x in range(41):
                t = self.color_chips.loc[(self.color_chips['H'] == x) & (self.color_chips['V'] == y)]
                if len(t) == 0:
                    s = ''
                elif len(t) == 1:
                    s = f(t)
                else:
                    raise TabError()

                print(s.ljust(pad), end="")
            print('')






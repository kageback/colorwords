import pickle

import matplotlib
import numpy as np

matplotlib.use('Agg')

from matplotlib import rc

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})

rc('text', usetex=True)

import matplotlib.pyplot as plt

from com_enviroments import wcs


def plot_with_std(exp, measure_id, x_id, z_id, measure_label=None, x_label=None, z_label=None):
    if measure_label is None:
        measure_label = measure_id.replace('_', ' ')
    if x_label is None:
        x_label = x_id.replace('_', ' ')
    if z_label is None:
        z_label = z_id.replace('_', ' ')

    y_avg = exp.get_reduced(measure_id, keep_axes_named=[x_id, z_id], reduce_method='avg')
    y_std = exp.get_reduced(measure_id, keep_axes_named=[x_id, z_id], reduce_method='std')

    x = exp.param_ranges[x_id]
    z = exp.param_ranges[z_id]

    fig, ax = plt.subplots()
    for z_i, z_value in enumerate(z):

        l = y_avg[:,z_i]
        std_l = y_std[:, z_i] / 4

        ax.plot(x, l,  '.', label=z_label + '=' + str(z_value))
        ax.fill_between(x, l - std_l, l + std_l, alpha=0.2)

    ax.legend()
    plt.ylabel(measure_label)
    plt.xlabel(x_label)

    fig_name = exp.pipeline_path + '/fig_' + measure_id + '_vs_' + x_id + '_for_' +  z_id +'.png'
    plt.savefig(fig_name)

def hist(exp, measure_id, x_id):
    dists = exp.reshape(measure_id, [x_id])
    x_vals = exp.param_ranges[x_id]
    np.savetxt(exp.pipeline_path + 'dists.txt', dists, delimiter=',', fmt='%u')
    np.savetxt(exp.pipeline_path + 'x_vals.txt', np.array(x_vals), delimiter=',', fmt='%u')
    for i in range(len(x_vals)):
        plt.figure()
        plt.hist(dists[i])
        fig_name = exp.pipeline_path + '/fig_hist_' + measure_id + '_vs_' + x_id + '-' + str(x_vals[i]) + '.png'
        plt.savefig(fig_name)

def plot_with_conf(exp, measure_id, x_id, measure_label=None, x_label=None, fmt='-'):
    if measure_label is None:
        measure_label = measure_id.replace('_', ' ')
    if x_label is None:
        x_label = x_id.replace('_', ' ')

    mean, ci = exp.estimate_mean(measure_id, as_function_of_axes=[x_id])

    x = exp.param_ranges[x_id]

    fig, ax = plt.subplots()
    ax.plot(x, mean,  fmt)
    ax.fill_between(x, ci[0], ci[1], alpha=0.2)

    plt.ylabel(measure_label)
    plt.xlabel(x_label)

    fig_name = exp.pipeline_path + '/fig_' + measure_id + '_vs_' + x_id +'.png'
    plt.savefig(fig_name)

def plot_lines_with_conf(exp, measure_id, x_id, z_id, measure_label=None, x_label=None, z_label=None, fmt='-'):
    if measure_label is None:
        measure_label = measure_id.replace('_', ' ')
    if x_label is None:
        x_label = x_id.replace('_', ' ')
    if z_label is None:
        z_label = z_id.replace('_', ' ')

    mean, ci = exp.estimate_mean(measure_id, as_function_of_axes=[x_id, z_id])

    x = exp.param_ranges[x_id]
    z = exp.param_ranges[z_id]

    fig, ax = plt.subplots()
    for z_i, z_value in enumerate(z):
        ax.plot(x, mean[:, z_i],  fmt, label=z_label + '= {0:.1f}'.format(z_value))
        ax.fill_between(x, ci[0][:, z_i], ci[1][:, z_i], alpha=0.2)

    ax.legend()
    plt.ylabel(measure_label)
    plt.xlabel(x_label)

    fig_name = exp.pipeline_path + '/fig_' + measure_id + '_vs_' + x_id + '_for_' +  z_id +'.png'
    plt.savefig(fig_name)

def plot_with_conf2(exp, measure_id, group_by_measure_id, z_id, measure_label=None, group_by_measure_label=None, z_label=None, ylim=None, xlim=None, fmt='-'):
    if measure_label is None:
        measure_label = measure_id.replace('_', ' ')
    if group_by_measure_label is None:
        group_by_measure_label = group_by_measure_id.replace('_', ' ')
    if z_label is None:
        z_label = z_id.replace('_', ' ')

    measure = exp.reshape(measure_id, as_function_of_axes=[z_id])
    group_by_measure = exp.reshape(group_by_measure_id, as_function_of_axes=[z_id])

    x = np.unique(group_by_measure)
    z = exp.param_ranges[z_id]


    fig, ax = plt.subplots()
    for z_i, z_value in enumerate(z):
        means = []
        cis = []
        for t in x:
            mean, ci = estimate_mean(measure[z_i][group_by_measure[z_i] == t])
            means += [mean]
            cis += [np.array(ci)]
        means = np.array(means)
        cis = np.array(cis)
        ax.plot(x, means, fmt, label=z_label + '= {0:.1f}'.format(z_value))
        ax.fill_between(x, cis[:, 0], cis[:, 1], alpha=0.2)

    ax.legend()
    plt.ylabel(measure_label)
    plt.xlabel(group_by_measure_label)
    if not xlim is None:
        plt.xlim(xlim)
    if not ylim is None:
        plt.ylim(ylim)
    fig_name = exp.pipeline_path + '/fig_' + measure_id + '_vs_' + group_by_measure_id + '_for_' +  z_id +'.png'
    plt.savefig(fig_name)

from scipy.stats import t
def estimate_mean(x, confidence_interval=0.95):
    n = x.shape[-1]

    mean = x.mean(axis=-1)
    stddev = x.std(axis=-1, ddof=1)  # Sample variance

    # Get the endpoints of the range that contains 95% of the distribution
    # The degree used in calculations is N - ddof
    t_bounds = t.interval(confidence_interval, n - 1)
    ci = [mean + c * stddev / np.sqrt(n) for c in t_bounds]
    return mean, ci

from gridengine.pipeline import Experiment
def main():
    exp = Experiment.load('color_fix.10')
    plot_with_conf2(exp, 'regier_cost', 'term_usage', 'com_noise',z_label='com $\sigma^2$')
    plot_with_conf2(exp, 'gibson_cost', 'term_usage', 'com_noise')
    plot_with_conf2(exp, 'wellformedness', 'term_usage', 'com_noise')



if __name__ == "__main__":
    main()


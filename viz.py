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


def plot_result(exp, measure_id, x_id, z_id, measure_label=None, x_label=None, z_label=None):
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
    d = exp.reshape(measure_id, [x_id])
    plt.figure()
    plt.hist(d)
    fig_name = exp.pipeline_path + '/fig_hist_' + measure_id + '_vs_' + x_id + '.png'
    plt.savefig(fig_name)


def plot_with_conf(exp, measure_id, x_id, z_id, measure_label=None, x_label=None, z_label=None, fmt='-'):
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


def plot_colormap(pipeline, taskid, plot_file_name):
    # Load results
    res_path = pipeline.pipeline_path + '/task.' + str(taskid) + '.result.pkl'
    with open(res_path, 'rb') as f:
        res = pickle.load(f)
    V = {k: res['V'][k]['word'] for k in range(330)}
    wcs.plot_with_colors(V, pipeline.pipeline_path + '/' + plot_file_name + '.png')


def plot_task_range(pipeline, start_task, range_name=''):
    num_of_words = range(3, 12)
    for taskid, nwords in zip(range(start_task, start_task + len(num_of_words)), num_of_words):
        plot_file_name = 'fig_colormap_' + range_name + '_' + '_nwords' + str(nwords) + '_' + pipeline.pipeline_name.replace('.', '') + '_task' + str(taskid)
        plot_colormap(pipeline, taskid, plot_file_name)


from gridengine.pipeline import Experiment
def main():
    exp = Experiment.load('num_symbs.0')
    #plot_com_noise_cost(exp)
    # wcs.plot_with_colors(wcs.language_map(32), 'Culina.png')
    # wcs.plot_with_colors(wcs.language_map(36), 'Ejagam.png')
    # wcs.plot_with_colors(wcs.language_map(47), 'iduna.png')
    # wcs.plot_with_colors(wcs.language_map(16), 'Buglere.png')
    #
    # job_id = 'avg50.0'
    # job = ge.Job(job_id=job_id, load_existing_job=True)
    #plot_costs(job)

    # plot color maps
    #plot_colormap(job, 350, 'fig_colormap_dev2')
    # no noise different #words
    # start_task=0 => noise = 0
    # start_task=342 => noise = 25

    #plot_task_range(job, 0, 'noise0')
    #plot_task_range(job, 342, 'noise25')
    #plot_task_range(job, 495,'noise50')
    #plot_task_range(job, 603,'noise100')







if __name__ == "__main__":
    main()


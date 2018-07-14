import pickle

import matplotlib
import numpy as np

matplotlib.use('Agg')

from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import matplotlib.pyplot as plt

from com_enviroments import wcs


def plot_result(exp, measure_id, x_id, z_id, measure_label=None, x_label=None, z_label=None, task_result_index=0):
    if measure_label is None:
        measure_label = measure_id.replace('_', ' ')
    if x_label is None:
        x_label = x_id.replace('_', ' ')
    if z_label is None:
        z_label = z_id.replace('_', ' ')

    y_avg = exp.get_reduced(measure_id, task_result_index=task_result_index, keep_axes_named=[x_id, z_id], reduce_method='avg')
    y_std = exp.get_reduced(measure_id, task_result_index=task_result_index, keep_axes_named=[x_id, z_id], reduce_method='std')

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


def print_ranges(exp, env):
    pass

def plot_com_noise_cost(exp):

    gibson_cost_avg = exp.get_reduced('gibson_cost', task_result_index=1, keep_axes_named=['msg_dim', 'com_noise'], reduce_method='avg')
    gibson_cost_std = exp.get_reduced('gibson_cost', task_result_index=1, keep_axes_named=['msg_dim', 'com_noise'], reduce_method='std')

    com_noise = exp.param_ranges['com_noise']
    msg_dim = exp.param_ranges['msg_dim']

    fig, ax = plt.subplots()
    for msg_dim_i, msg_dim_value in enumerate(msg_dim):

        l = gibson_cost_avg[msg_dim_i, :]
        std_l = gibson_cost_std[msg_dim_i, :] / 4

        ax.plot(com_noise, l,  '.', label='terms=' + str(msg_dim_value))
        ax.fill_between(com_noise, l - std_l, l + std_l, alpha=0.2)

    ax.legend()
    plt.xlabel('Communication noise')
    plt.ylabel('Gibson communication efficiency')

    fig_name = exp.pipeline_path + '/fig_gibson_vs_com_noise.png'
    plt.savefig(fig_name)
    

def plot_reiger_gibson(exp):
    regier_cost_avg = exp.get_reduced('regier_cost', keep_axes_named=['msg_dim', 'perception_noise'], reduce_method='avg')
    regier_cost_std = exp.get_reduced('regier_cost', keep_axes_named=['msg_dim', 'perception_noise'], reduce_method='std')

    gibson_cost_avg = exp.get_reduced('gibson_cost', task_result_index=1, keep_axes_named=['msg_dim', 'perception_noise'], reduce_method='avg')
    gibson_cost_std = exp.get_reduced('gibson_cost', task_result_index=1, keep_axes_named=['msg_dim', 'perception_noise'], reduce_method='std')

    noise_values = exp.param_ranges['perception_noise']
    msg_dim_values = exp.param_ranges['msg_dim']

    # Plot regier and gibson_cost
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

    for noise_i, noise_value in enumerate(noise_values):
        l = regier_cost_avg[:, noise_i]
        std_l = regier_cost_std[:, noise_i] / 4
        ax[0].plot(msg_dim_values, l,  '.', label='$\sigma^2=' + str(noise_value) + '$')
        ax[0].fill_between(msg_dim_values, l - std_l, l + std_l, alpha=0.2)
    ax[0].legend()
    ax[0].set_title('Communication cost (bits)')

    for noise_i, noise_value in enumerate(noise_values):
        l = gibson_cost_avg[:, noise_i]
        std_l = gibson_cost_std[:, noise_i] / 4
        ax[1].plot(msg_dim_values, l, '.', label='$\sigma^2=' + str(noise_value) + '$')
        ax[1].fill_between(msg_dim_values, l - std_l, l + std_l, alpha=0.2)
    ax[1].legend()


    ax[1].set_title('Communication efficiency (bits)')

    fig.text(0.04, 0.5, 'Cost/efficiency', va='center', rotation='vertical')
    fig.text(0.5, 0.04, 'Number of color words', ha='center')

    fig.subplots_adjust(wspace=0)
    plt.setp([a.get_yticklabels() for a in fig.axes[1:]], visible=False)


    fig_name = exp.pipeline_path + '/fig_reiger_gibson.png'
    plt.savefig(fig_name)


def plot_wellformedness(exp):
    wellformedness_avg = exp.get_reduced('wellformedness', keep_axes_named=['msg_dim', 'perception_noise'], reduce_method='avg')
    wellformedness_std = exp.get_reduced('wellformedness', keep_axes_named=['msg_dim', 'perception_noise'], reduce_method='std')

    noise_values = exp.param_ranges['perception_noise']
    msg_dim_values = exp.param_ranges['msg_dim']

    fig, ax = plt.subplots()
    for noise_i, noise_value in enumerate(noise_values):
        l = wellformedness_avg[:, noise_i]
        std_l = wellformedness_std[:, noise_i] / 4
        ax.plot(msg_dim_values, l,  '.', label='$\sigma^2=' + str(noise_value) + '$')
        ax.fill_between(msg_dim_values, l - std_l, l + std_l, alpha=0.2)
    ax.legend()
    plt.xlabel('Number of color words')
    plt.ylabel('Wellformedness')

    fig_name = exp.pipeline_path + '/fig_wellformedness.png'
    plt.savefig(fig_name)


def plot_term_usage(exp):
    # plot term usage for all #words
    term_usage_avg = exp.get_reduced('term_usage', keep_axes_named=['msg_dim', 'perception_noise'], reduce_method='avg')
    term_usage_std = exp.get_reduced('term_usage', keep_axes_named=['msg_dim', 'perception_noise'], reduce_method='std')
    #term_usage = exp.to_numpy('term_usage')
    noise_values = exp.param_ranges['perception_noise']
    msg_dim_values = exp.param_ranges['msg_dim']

    #noise_values = [noise_values[0]] + noise_values[2:-1]
    index = np.arange(len(msg_dim_values))
    fig, ax = plt.subplots()
    # dim_values_used = [5,6,7,8,9,10,11]
    for msg_dim_i, msg_dim_value in enumerate(msg_dim_values):
        l = term_usage_avg[msg_dim_i, :]
        std_l = term_usage_std[msg_dim_i, :] / 4

        ax.plot(noise_values, l, '.', label='terms$\leq$' + str(msg_dim_value))
        ax.fill_between(noise_values, l - std_l, l + std_l, alpha=0.2)
    ax.legend()
    plt.xlabel('Noise variance ($\sigma^2$)')
    plt.ylabel('Average number of color terms')
    plt.xticks(noise_values)
    plt.ylim([4, 9])

    fig_name = exp.pipeline_path + '/fig_color_term_usage_all.png'
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
    plot_com_noise_cost(exp)
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


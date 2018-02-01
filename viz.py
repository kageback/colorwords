import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import matplotlib.pyplot as plt
import gridengine.batch as ge

import wcs


def plot_costs(job):

    # Load results
    res_path = job.job_dir + '/result.pkl'
    with open(res_path, 'rb') as f:
        res = pickle.load(f)

    noise_values = res['noise_values']
    msg_dim_values = res['msg_dim_values']

    regier_cost = res['regier_cost']
    wellformedness = res['wellformedness']
    combined_criterion = res['combined_criterion']
    term_usage = res['term_usage']
    gibson_cost = res['gibson_cost']
    avg_over = res['avg_over']

    # Plot gibson_cost
    fig, ax = plt.subplots()
    for noise_value in noise_values:
        l = []
        std_l = []
        for msg_dim_value in msg_dim_values:
            l.append(regier_cost[(noise_value, msg_dim_value)]['mean'])
            std_l.append(np.sqrt(regier_cost[(noise_value, msg_dim_value)]['var']))
        l = np.array(l)
        std_l = np.array(std_l) / 4
        ax.plot(msg_dim_values, l, '.-', label='$\sigma=' + str(noise_value) + '$')
        ax.fill_between(msg_dim_values, l - std_l, l + std_l, alpha=0.2)
    ax.legend()
    plt.xlabel('Number of color words')
    plt.ylabel('Communication cost')

    fig_name = job.job_dir + '/fig_gibson_cost.png'
    plt.savefig(fig_name)

    # Plot reiger_cost
    fig, ax = plt.subplots()
    for noise_value in noise_values:
        l = []
        std_l = []
        for msg_dim_value in msg_dim_values:
            l.append(regier_cost[(noise_value, msg_dim_value)]['mean'])
            std_l.append(np.sqrt(regier_cost[(noise_value, msg_dim_value)]['var']))
        l = np.array(l)
        std_l = np.array(std_l) / 4
        ax.plot(msg_dim_values, l,  '.-' ,label='$\sigma=' + str(noise_value) + '$')
        ax.fill_between(msg_dim_values, l - std_l, l + std_l, alpha=0.2)
    ax.legend()
    plt.xlabel('Number of color words')
    plt.ylabel('Communication cost')

    fig_name = job.job_dir + '/fig_commcost.png'
    plt.savefig(fig_name)


    # plot wellformedness
    fig, ax = plt.subplots()
    for noise_value in noise_values:
        l = []
        std_l = []
        for msg_dim_value in msg_dim_values:
            l.append(wellformedness[(noise_value, msg_dim_value)]['mean'])
            std_l.append(np.sqrt(wellformedness[(noise_value, msg_dim_value)]['var']))
        l = np.array(l)
        std_l = np.array(std_l) / 4
        ax.plot(msg_dim_values, l,  '.-' ,label='$\sigma=' + str(noise_value) + '$')
        ax.fill_between(msg_dim_values, l - std_l, l + std_l, alpha=0.2)
    ax.legend()
    plt.xlabel('Number of color words')
    plt.ylabel('Wellformedness')

    fig_name = job.job_dir + '/fig_wellformedness.png'
    plt.savefig(fig_name)


    # plot combined_criterion
    fig, ax = plt.subplots()
    for noise_value in noise_values:
        l = []
        std_l = []
        for msg_dim_value in msg_dim_values:
            l.append(combined_criterion[(noise_value, msg_dim_value)]['mean'])
            std_l.append(np.sqrt(combined_criterion[(noise_value, msg_dim_value)]['var']))
        l = np.array(l)
        std_l = np.array(std_l) / 4
        ax.plot(msg_dim_values, l, '.-' ,label='$\sigma=' + str(noise_value) + '$')
        ax.fill_between(msg_dim_values, l - std_l, l + std_l, alpha=0.2)
    ax.legend()
    plt.xlabel('Number of color words')
    plt.ylabel('Combined criterion')

    fig_name = job.job_dir + '/fig_combined_criterion.png'
    plt.savefig(fig_name)


    # plot term usage
    fig, ax = plt.subplots()
    msg_dim_value = 11
    l = []
    std_l = []
    for noise_value in noise_values:
        l.append(term_usage[(noise_value, msg_dim_value)]['mean'])
        std_l.append(np.sqrt(term_usage[(noise_value, msg_dim_value)]['var']))
    l = np.array(l)
    std_l = np.array(std_l)
    #ax.plot(noise_values, l, '.', label='$\sigma=' + str(noise_value) + '$')
    ax.fill_between(noise_values, l - std_l, l + std_l, alpha=0.2)

    #ax.bar(noise_values, l, width=10, yerr=std_l, ecolor='k', capsize=5)
    ax.errorbar(noise_values, l, yerr=std_l, fmt='o', ecolor='g',  capsize=5)

    #ax.legend()
    plt.xlabel('Noise variance')
    plt.ylabel('Average number of color terms')
    plt.xticks([0,25,50,100])
    plt.ylim([5,10])

    fig_name = job.job_dir + '/fig_color_term_usage_' + str(msg_dim_value) + '.png'
    plt.savefig(fig_name)


    # plot term usage for all #words
    index = np.arange(len(msg_dim_values))
    bar_width = 0.35
    opacity = 0.8
    fig, ax = plt.subplots()
    dim_values_used = [6,7,8,9,10,11]
    for msg_dim_value, i in zip(dim_values_used, range(len(dim_values_used))):
        l = []
        std_l = []
        for noise_value in noise_values:
            l.append(term_usage[(noise_value, msg_dim_value)]['mean'])
            std_l.append(np.sqrt(term_usage[(noise_value, msg_dim_value)]['var']))
        l = np.array(l)
        std_l = np.array(std_l) / 4
        ax.plot(noise_values, l, '.', label='terms$\leq$' + str(msg_dim_value))
        ax.fill_between(noise_values, l - std_l, l + std_l, alpha=0.2)

        #ax.errorbar(noise_values, l, label=str(msg_dim_value), yerr=std_l, fmt='o',  capsize=5)
        #ax.bar(noise_values, l, width=10)
        #plt.bar(noise_values + i*bar_width, l, bar_width,
        #        alpha=opacity,
        #        label=str(msg_dim_value))
    ax.legend()
    plt.xlabel('Noise variance')
    plt.ylabel('Average number of color terms')
    plt.xticks([0, 25, 50, 100])
    plt.ylim([5, 9])

    fig_name = job.job_dir + '/fig_color_term_usage_all.png'
    plt.savefig(fig_name)


def plot_colormap(job, taskid, plot_file_name):
    # Load results
    res_path = job.job_dir + '/task.' + str(taskid) + '.result.pkl'
    with open(res_path, 'rb') as f:
        res = pickle.load(f)

    wcs.plot_with_colors(res['V'], job.job_dir + '/' + plot_file_name + '.png')


def plot_task_range(job, start_task, range_name=''):

    num_of_words = range(3, 12)

    for taskid, nwords in zip(range(start_task, start_task + len(num_of_words)), num_of_words):

        plot_file_name = 'fig_colormap_' + range_name + '_' + '_nwords' + str(nwords) + '_' + job.job_id.replace('.', '') + '_task' + str(taskid)

        plot_colormap(job, taskid, plot_file_name)


def main():
    job_id = 'gibson.0'
    job = ge.Job(job_id=job_id, load_existing_job=True)
    plot_costs(job)

    # plot color maps
    #plot_colormap(job, 350, 'fig_colormap_dev')
    # no noise different #words
    # start_task=0 => noise = 0
    # start_task=342 => noise = 25

    #plot_task_range(job, 0, 'noise0')
    #plot_task_range(job, 342, 'noise25')
    #plot_task_range(job, 495,'noise50')
    #plot_task_range(job, 603,'noise100')







if __name__ == "__main__":
    main()


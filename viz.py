import pickle
import matplotlib
matplotlib.use('Agg')
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
    avg_over = res['avg_over']

    # Plot commcost
    fig, ax = plt.subplots()
    for noise_value in noise_values:
        l = []
        for msg_dim_value in msg_dim_values:
            l.append(regier_cost[(noise_value, msg_dim_value)] / avg_over[(noise_value, msg_dim_value)])
        ax.plot(msg_dim_values, l, label=str(noise_value))
    ax.legend()
    plt.xlabel('Number of color words')
    plt.ylabel('Communication cost')

    fig_name = job.job_dir + '/fig_commcost.png'
    plt.savefig(fig_name)


    # plot wellformedness
    fig, ax = plt.subplots()
    for noise_value in noise_values:
        l = []
        for msg_dim_value in msg_dim_values:
            l.append(wellformedness[(noise_value, msg_dim_value)] / avg_over[(noise_value, msg_dim_value)])
        ax.plot(msg_dim_values, l, label=str(noise_value))
    ax.legend()
    plt.xlabel('Number of color words')
    plt.ylabel('Wellformedness')

    fig_name = job.job_dir + '/fig_wellformedness.png'
    plt.savefig(fig_name)


    # plot combined_criterion
    fig, ax = plt.subplots()
    for noise_value in noise_values:
        l = []
        for msg_dim_value in msg_dim_values:
            l.append(combined_criterion[(noise_value, msg_dim_value)] / avg_over[(noise_value, msg_dim_value)])
        ax.plot(msg_dim_values, l, label=str(noise_value))
    ax.legend()
    plt.xlabel('Number of color words')
    plt.ylabel('Combined criterion')

    fig_name = job.job_dir + '/fig_combined_criterion.png'
    plt.savefig(fig_name)


def plot_colormap(job, taskid, plot_file_name):
    # Load results
    res_path = job.job_dir + '/task.' + str(taskid) + '.result.pkl'
    with open(res_path, 'rb') as f:
        res = pickle.load(f)

    wcs.plot_with_colors(res['V'], job.job_dir + '/' + plot_file_name + '.png',
                         y_wcs_range=' ABCDEFGHIJ ', x_wcs_range=range(0, 41),
                         use_real_color=True, add_boarders_color='w')


def plot_task_range(job, start_task, range_name=''):

    num_of_words = range(3, 12)

    for taskid, nwords in zip(range(start_task, start_task + len(num_of_words)), num_of_words):

        plot_file_name = 'fig_colormap_' + range_name + '_' + '_nwords' + str(nwords) + '_' + job.job_id.replace('.', '') + '_task' + str(taskid)

        plot_colormap(job, taskid, plot_file_name)


def main():
    job_id = 'job.16'
    job = ge.Job(job_id=job_id, load_existing_job=True)
    #plot_costs(job)

    # plot color maps
    # no noise different #words
    # start_task=0 => noise = 0
    # start_task=342 => noise = 25

    plot_task_range(job, 0, 'noise0')
    #plot_task_range(job, 342, 'noise25')
    #plot_task_range(job, 495,'noise50')
    #plot_task_range(job, 603,'noise100')





if __name__ == "__main__":
    main()


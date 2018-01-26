import os
import pickle
import gridengine.batch as ge

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wcs


def map_job(job, args):
    for noise_i, noise in zip(range(len(args['noise_range'])), args['noise_range']):
        for avg_i in range(args['avg_over']):
            for i, dim in zip(range(len(args['msg_dim_range'])), args['msg_dim_range']):
                job.run_python_script(args['script'],
                                      ge_gpu=0,
                                      msg_dim=dim,
                                      max_epochs=args['max_epochs'],
                                      noise_level=noise)


def reduce_job(job):

    def inc_dict(dict, key, increment):
        if key in dict.keys():
            dict[key] += increment
        else:
            dict[key] = increment

    taskid = 0
    noise_values = []
    msg_dim_values = []
    avg_over = {}
    regier_cost = {}
    min_k_cut_cost = {}
    while True:
        res_path = job.job_dir + '/' + ge.get_task_name(taskid) + '.result.pkl'
        if not os.path.isfile(res_path):
            break
        print('\r Reducing results and computing evaluation metrics for ' + ge.get_task_name(taskid), end='')
        with open(res_path, 'rb') as f:

            # read data
            task_res = pickle.load(f)
            noise_level = task_res['args'].noise_level
            msg_dim = task_res['args'].msg_dim
            V = task_res['V']

            # compute error measures
            inc_dict(regier_cost, (noise_level, msg_dim), wcs.communication_cost_regier(V))  # task_res['regier_cost']
            inc_dict(min_k_cut_cost, (noise_level, msg_dim),
                     wcs.min_k_cut_cost(V, msg_dim))  # task_res['min_k_cut_cost'] or wcs.min_k_cut_cost(V, msg_dim)
            inc_dict(avg_over, (noise_level, msg_dim), 1)

            # Compile implicit argument ranges for plotting
            if noise_level not in noise_values:
                noise_values.append(noise_level)
            if msg_dim not in msg_dim_values:
                msg_dim_values.append(msg_dim)

        taskid += 1

    # Sorting might not be necessary since tasks are created in order but what the heck...
    noise_values.sort()
    msg_dim_values.sort()
    print('\n Reduce done!')

    # Save result
    res = {}
    res['noise_values'] = noise_values
    res['msg_dim_values'] = msg_dim_values

    res['regier_cost'] = regier_cost
    res['min_k_cut_cost'] = min_k_cut_cost
    res['avg_over'] = avg_over

    res_path = job.job_dir + '/result.pkl'
    print('saving result as', res_path)
    with open(res_path, 'wb') as f:
        pickle.dump(res, f)


def plot(job):

    # Load results
    res_path = job.job_dir + '/result.pkl'
    with open(res_path, 'rb') as f:
        res = pickle.load(f)

    noise_values = res['noise_values']
    msg_dim_values = res['msg_dim_values']

    regier_cost = res['regier_cost']
    min_k_cut_cost = res['min_k_cut_cost']
    avg_over = res['avg_over']


    # Plot results
    fig, ax = plt.subplots()
    for noise_value in noise_values:
        l = []
        for msg_dim_value in msg_dim_values:
            l.append(regier_cost[(noise_value, msg_dim_value)] / avg_over[(noise_value, msg_dim_value)])
        ax.plot(msg_dim_values, l, label=str(noise_value))
    ax.legend()

    fig_name = job.job_dir + '/commcost.png'
    plt.savefig(fig_name)

    fig, ax = plt.subplots()
    for noise_value in noise_values:
        l = []
        for msg_dim_value in msg_dim_values:
            l.append(min_k_cut_cost[(noise_value, msg_dim_value)] / avg_over[(noise_value, msg_dim_value)])
        ax.plot(msg_dim_values, l, label=str(noise_value))
    ax.legend()

    fig_name = job.job_dir + '/cutcost.png'
    plt.savefig(fig_name)


def main():
    num_words_min = 3
    num_words_max = 4

    args = {}
    args['script'] = 'model.py'
    args['max_epochs'] = 100 #10000
    args['noise_range'] = [0, 25]#, 50, 100]
    args['avg_over'] = 2
    args['msg_dim_range'] = range(num_words_min, num_words_max+1)
    job = ge.Job()
    args['job_id'] = job.job_id

    args_path = job.job_dir + '/job_args.pkl'
    print('saving job arguments as', args_path)
    with open(args_path, 'wb') as f:
        pickle.dump(args, f)

    print('Map job')
    map_job(job, args)

    print('Waiting for tasks to finish...')
    job.wait()

    print('Reduce job')
    reduce_job(job)

    print('plot results')
    plot(job)


if __name__ == "__main__":
    main()

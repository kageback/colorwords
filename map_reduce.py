import os
import pickle
import gridengine.batch as ge
import torchHelpers as th
import torch.nn.functional as F
import numpy as np

import wcs

# for a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count = count + 1
    delta = newValue - mean
    mean = mean + delta / count
    delta2 = newValue - mean
    M2 = M2 + delta * delta2

    return count, mean, M2

# retrieve the mean and variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return count, mean, float('nan')

    return count, mean, M2/(count - 1)


def update_stats_dict(d, key, new_value):
    if key not in d.keys():
        d[key] = tuple([0, 0, 0])

    d[key] = update(d[key], new_value)


def finalize_stats_dict(d):
    for k in d.keys():
        stats = finalize(d[k])
        d[k] = {'count': stats[0], 'mean': stats[1], 'var': stats[2]}

    return d


def compute_gibson_cost(a):
    chip_indices, colors = wcs.all_colors()
    colors = th.float_var(colors, False)
    color_terms = th.long_var(range(a.msg_dim), False)

    p_WC = a(perception=colors).t().data.numpy()
    p_CW = F.softmax(a(msg=color_terms), dim=1).data.numpy()

    S = -np.diag(np.matmul(p_WC.transpose(), (np.log2(p_CW))))

    avg_S = S.sum() / len(S)  # expectation assuming uniform prior


    # debug code
    # s = 0
    # c = 43
    # for w in range(a.msg_dim):
    #     s += -p_WC[w, c]*np.log2(p_CW[w, c])
    # print(S[c] - s)

    return S, avg_S


def map_job(job, args):
    for avg_i in range(args['avg_over']):
        for noise_i, noise in zip(range(len(args['noise_range'])), args['noise_range']):
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
    wellformedness = {}
    combined_criterion = {}
    term_usage = {}
    gibson_cost = {}

    while True:
        res_path = job.job_dir + '/' + ge.get_task_name(taskid) + '.result.pkl'
        if not os.path.isfile(res_path):
            break
        print('\r Reducing results and computing evaluation metrics for ' + job.job_id + ' ' + ge.get_task_name(taskid), end='')
        with open(res_path, 'rb') as f:

            # read data
            task_res = pickle.load(f)
            noise_level = task_res['args'].noise_level
            msg_dim = task_res['args'].msg_dim
            V = task_res['V']



            #### test agents
            # a = task_res['agent']
            # chip_indices, colors = wcs.all_colors()
            # colors = th.float_var(colors, False)
            #
            # probs = a(perception=colors)
            # m = Categorical(probs)
            # msg = m.sample()
            #
            # color_guess = a(msg=msg)
            ###

            # compute error measures
            #inc_dict(regier_cost_old, (noise_level, msg_dim), wcs.communication_cost_regier(V))
            #inc_dict(wellformedness, (noise_level, msg_dim), wcs.wellformedness(V))
            #inc_dict(combined_criterion, (noise_level, msg_dim), wcs.combined_criterion(V))
            #inc_dict(term_usage, (noise_level, msg_dim), wcs.compute_term_usage(V)[0])
            inc_dict(avg_over, (noise_level, msg_dim), 1)

            update_stats_dict(regier_cost, (noise_level, msg_dim), wcs.communication_cost_regier(V))
            update_stats_dict(wellformedness, (noise_level, msg_dim), wcs.wellformedness(V))
            update_stats_dict(combined_criterion, (noise_level, msg_dim), wcs.combined_criterion(V))
            update_stats_dict(term_usage, (noise_level, msg_dim), wcs.compute_term_usage(V)[0])
            update_stats_dict(gibson_cost, (noise_level, msg_dim), compute_gibson_cost(task_res['agent'])[1])


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

    res['regier_cost'] = finalize_stats_dict(regier_cost)
    res['wellformedness'] = finalize_stats_dict(wellformedness)
    res['combined_criterion'] = finalize_stats_dict(combined_criterion)
    res['term_usage'] = finalize_stats_dict(term_usage)
    res['gibson_cost'] = finalize_stats_dict(gibson_cost)
    res['avg_over'] = avg_over


    res_path = job.job_dir + '/result.pkl'
    print('saving result as', res_path)
    with open(res_path, 'wb') as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    job_id = 'gibson.0'
    job = ge.Job(job_id=job_id, load_existing_job=True)
    reduce_job(job)

import os
import pickle
import gridengine.batch as ge

import wcs


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
            inc_dict(regier_cost, (noise_level, msg_dim), wcs.communication_cost_regier(V))
            inc_dict(wellformedness, (noise_level, msg_dim), wcs.wellformedness(V))
            inc_dict(combined_criterion, (noise_level, msg_dim), wcs.combined_criterion(V))
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
    res['wellformedness'] = wellformedness
    res['combined_criterion'] = combined_criterion
    res['avg_over'] = avg_over

    res_path = job.job_dir + '/result.pkl'
    print('saving result as', res_path)
    with open(res_path, 'wb') as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    job_id = 'job.3'
    job = ge.Job(job_id=job_id, load_existing_job=True)
    reduce_job(job)

import pickle
import gridengine.batch as ge
import viz
import map_reduce


def main():
    num_words_min = 3
    num_words_max = 11

    args = {}
    args['script'] = 'model.py'
    args['max_epochs'] = 10000 #10000
    args['noise_range'] = [25]#0, 25, 50, 100]
    args['avg_over'] = 20
    args['msg_dim_range'] = range(num_words_min, num_words_max+1)
    job = ge.Job(job_id='dev')
    args['job_id'] = job.job_id

    args_path = job.job_dir + '/job_args.pkl'
    print('saving job arguments as', args_path)
    with open(args_path, 'wb') as f:
        pickle.dump(args, f)

    print('Map job')
    map_reduce.map_job(job, args)

    print('Waiting for tasks to finish...')
    job.wait()

    print('Reduce job')
    map_reduce.reduce_job(job)

    #print('plot results')
    #viz.plot_costs(job)


if __name__ == "__main__":
    main()

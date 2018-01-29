import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gridengine.batch as ge


def plot(job):

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


if __name__ == "__main__":
    job_id = 'job.3'

    job = ge.Job(job_id=job_id, load_existing_job=True)
    plot(job)

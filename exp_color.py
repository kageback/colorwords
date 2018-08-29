import numpy as np
import viz
import matplotlib.pyplot as plt
import gridengine as sge

import evaluate
from gridengine.pipeline import Experiment
import com_enviroments
import exp_shared

import exp_color_rl
import exp_color_cielab_cc
import exp_color_random
import Correlation_Clustering

def run(host_name, pipeline='', exp_rl_id='', exp_ccc_id='', exp_random_id=''):
    if pipeline != '':
        return exp_shared.load_exp(pipeline)

    exp = Experiment(exp_name='color_dev',
                     fixed_params=[('exp_rl_id', exp_rl_id),
                                   ('exp_ccc_id', exp_ccc_id),
                                   ('exp_random_id', exp_random_id)])

    # RL experiment
    exp_rl = exp_color_rl.run(host_name, pipeline=exp.fixed_params['exp_rl_id'])
    exp.set_result('exp_rl', value=exp_rl)

    # cielab correlation clustering experiment
    exp_ccc = exp_color_cielab_cc.run(host_name, pipeline=exp.fixed_params['exp_ccc_id'])
    exp.set_result('exp_ccc', value=exp_ccc)


    # random baseline experiment
    exp_random = exp_color_random.run(host_name, pipeline=exp.fixed_params['exp_random_id'])
    exp.set_result('exp_random', value=exp_random)

    exp.save()

    return exp


def visualize(exp):

    cost_plot(exp, 'regier_cost')
    cost_plot(exp, 'wellformedness')


def cost_plot(exp, measure_id):
    group_by_measure_id = 'term_usage'
    fig, ax = plt.subplots()
    add_line_to_axes(ax, exp.get('exp_rl'), measure_id, group_by_measure_id, line_label='RL')
    add_line_to_axes(ax, exp.get('exp_ccc'), measure_id, group_by_measure_id, line_label='CIELAB CC')
    add_line_to_axes(ax, exp.get('exp_random'), measure_id, group_by_measure_id, line_label='Random')
    ax.legend()
    measure_label = measure_id.replace('_', ' ')
    group_by_measure_label = group_by_measure_id.replace('_', ' ')
    plt.ylabel(measure_label)
    plt.xlabel(group_by_measure_label)
    plt.xlim([3,11])
    fig_name = exp.pipeline_path + '/fig_' + measure_id + '_vs_' + group_by_measure_id + '.png'
    plt.savefig(fig_name)


def add_line_to_axes(ax, exp_rl, measure_id, group_by_measure_id, line_label='RL', fmt='-'):
    measure = exp_rl.reshape(measure_id)
    group_by_measure = exp_rl.reshape(group_by_measure_id)
    x = np.unique(group_by_measure)
    means = []
    cis = []
    for t in x:
        mean, ci = viz.estimate_mean(measure[group_by_measure == t])
        means += [mean]
        cis += [np.array(ci)]
    means = np.array(means)
    cis = np.array(cis)
    ax.plot(x, means, fmt, label=line_label)
    ax.fill_between(x, cis[:, 0], cis[:, 1], alpha=0.2)


def com_plots(exp_rl):
    viz.plot_with_conf2(exp_rl, 'regier_cost', 'term_usage', 'com_noise', z_label='com $\sigma^2$')
    viz.plot_with_conf2(exp_rl, 'gibson_cost', 'term_usage', 'com_noise', z_label='com $\sigma^2$')
    viz.plot_with_conf2(exp_rl, 'wellformedness', 'term_usage', 'com_noise', z_label='com $\sigma^2$')


def main(args):
    # Run experiment
    exp = run(args.host_name,
              pipeline=args.pipeline,
              exp_rl_id=args.exp_rl_id,
              exp_ccc_id=args.exp_ccc_id,
              exp_random_id=args.exp_random_id)

    # Visualize experiment
    visualize(exp)

    # Todo regier_cost/gibson_cost/wellformedness per term usage including baselines

    #com_plots(exp.get('exp_ccc'))


if __name__ == "__main__":
    parser = exp_shared.parse_script_arguments()
    parser.add_argument('--exp_ccc_id', type=str, default='',
                        help='cielab correlation clustering experiment')
    parser.add_argument('--exp_rl_id', type=str, default='',
                        help='Reinforcement learning experiment')
    parser.add_argument('--exp_random_id', type=str, default='',
                        help='random baseline')
    main(parser.parse_args())

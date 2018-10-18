import numpy as np
import viz
import matplotlib.pyplot as plt
import gridengine as sge

import evaluate
from gridengine.pipeline import Experiment
import com_enviroments
import exp_shared

import exp_color_rl
import exp_color_ccc
import exp_color_random
import exp_color_human

import Correlation_Clustering

def run(host_name='local', pipeline='', exp_rl_id='', exp_ccc_id='', exp_random_id='', exp_human_id=''):
    if pipeline != '':
        return exp_shared.load_exp(pipeline)

    if exp_rl_id == '':
        exp_rl_id = exp_color_rl.run(host_name).pipeline_name

    if exp_ccc_id == '':
        exp_ccc_id = exp_color_ccc.run(host_name).pipeline_name

    if exp_random_id == '':
        exp_random_id = exp_color_random.run(host_name).pipeline_name

    if exp_human_id == '':
        exp_human_id = exp_color_human.run(host_name).pipeline_name

    exp = Experiment(exp_name='comb',
                     fixed_params=[('exp_rl_id', exp_rl_id),
                                   ('exp_ccc_id', exp_ccc_id),
                                   ('exp_random_id', exp_random_id),
                                   ('exp_human_id', exp_human_id)])
    exp.save()

    return exp


def visualize(exp):

    cost_plot(exp, 'regier_cost', ylabel='KL Loss', xlabel='Color terms used')
    cost_plot(exp, 'wellformedness', ylabel='Wellformedness', xlabel='Color terms used')


def cost_plot(exp, measure_id, ylabel='', xlabel='' ):
    group_by_measure_id = 'term_usage'
    fig, ax = plt.subplots()

    add_line_to_axes(ax, exp_color_ccc.run(pipeline=exp.fixed_params['exp_ccc_id']), measure_id, group_by_measure_id, line_label='CIELAB CC')
    add_line_to_axes(ax, exp_color_random.run(pipeline=exp.fixed_params['exp_random_id']), measure_id, group_by_measure_id, line_label='Random')
    add_scatter_to_axes(ax, exp_color_human.run(pipeline=exp.fixed_params['exp_human_id']), measure_id, group_by_measure_id, label='WCS languages')
    add_line_to_axes(ax, exp_color_human.run(pipeline=exp.fixed_params['exp_human_id']), measure_id, group_by_measure_id, line_label='WCS languages mean', conf=False)
    add_line_to_axes(ax, exp_color_rl.run(pipeline=exp.fixed_params['exp_rl_id']), measure_id, group_by_measure_id,
                     line_label='RL')

    ax.legend()
    if ylabel == '':
        ylabel = measure_id.replace('_', ' ')
    if xlabel == '':
        xlabel = group_by_measure_id.replace('_', ' ')

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim([3, 11])
    fig_name = exp.pipeline_path + '/fig_' + measure_id + '_vs_' + group_by_measure_id + '.png'
    plt.savefig(fig_name)

def add_scatter_to_axes(ax, exp_rl,  measure_id, group_by_measure_id, label=''):
    regier_cost = exp_rl.reshape(measure_id, as_function_of_axes=['lang_id'])
    term_usage = exp_rl.reshape(group_by_measure_id, as_function_of_axes=['lang_id'])
    ax.scatter(term_usage, regier_cost, label=label, alpha='0.3')

def add_line_to_axes(ax, exp_rl, measure_id, group_by_measure_id, line_label='RL', fmt='-', conf=True):
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
    if conf:
        ax.fill_between(x, cis[:, 0], cis[:, 1], alpha=0.2)


def main(args):
    # Run experiment
    exp = run(args.host_name,
              pipeline=args.pipeline,
              exp_rl_id=args.exp_rl_id,
              exp_ccc_id=args.exp_ccc_id,
              exp_random_id=args.exp_random_id,
              exp_human_id=args.exp_human_id)

    # Visualize experiment
    visualize(exp)



if __name__ == "__main__":
    parser = exp_shared.parse_script_arguments()
    parser.add_argument('--exp_ccc_id', type=str, default='',
                        help='cielab correlation clustering experiment')
    parser.add_argument('--exp_rl_id', type=str, default='',
                        help='Reinforcement learning experiment')
    parser.add_argument('--exp_random_id', type=str, default='',
                        help='random baseline')
    parser.add_argument('--exp_human_id', type=str, default='',
                        help='Human baseline')
    main(parser.parse_args())

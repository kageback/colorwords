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

    cost_plot(exp, 'regier_cost', ylabel='KL Loss', xlabel='Color terms used', xlim=[3, 11])
    cost_plot(exp, 'wellformedness', ylabel='Well-formedness', xlabel='Color terms used', ylim=[30000, 50000], xlim=[3, 11])


def cost_plot(exp, measure_id, ylabel='', xlabel='', ylim=None, xlim=None):
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
    if not xlim is None:
        plt.xlim(xlim)
    if not ylim is None:
        plt.ylim(ylim)

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


def print_tables(exp):
    term_usage_to_analyse = list(range(3, 12))

    exp_rl = exp_color_rl.run(pipeline=exp.fixed_params['exp_rl_id'])

    agent_maps = exp_rl.reshape('agent_language_map')
    agent_term_usage = exp_rl.reshape('term_usage')
    maps_vs_noise = exp_rl.reshape('agent_language_map', as_function_of_axes=['perception_noise'])
    term_usage_vs_noise = exp_rl.reshape('term_usage', as_function_of_axes=['perception_noise'])

    exp_human = exp_color_human.run(pipeline=exp.fixed_params['exp_human_id'])
    human_maps = exp_human.reshape('language_map')
    human_term_usage = exp_human.reshape('term_usage')  #np.array([np.unique(m).shape[0] for m in human_maps])

    exp_ccc = exp_color_human.run(pipeline=exp.fixed_params['exp_ccc_id'])
    ccc_maps = exp_ccc.reshape('language_map')
    ccc_term_usage = exp_ccc.reshape('term_usage')

    human_to_human = []

    agent_to_agent = []
    agent_to_agent_per_noise = []
    human_to_agent = []
    human_to_CCC = []
    agent_to_CCC = []
    human_to_random = []

    for t in term_usage_to_analyse:
        agent_to_agent += [evaluate.mean_rand_index(agent_maps[agent_term_usage == t])]

        a = np.array([evaluate.mean_rand_index(maps_vs_noise[noise_i][term_usage_vs_noise[noise_i] == t])
                      for noise_i in range(len(maps_vs_noise))])

        agent_to_agent_per_noise += [a[~np.isnan(a)].mean()]

        human_to_human += [evaluate.mean_rand_index(human_maps[human_term_usage == t])]

        human_to_agent += [evaluate.mean_rand_index(human_maps[human_term_usage == t],
                                                    agent_maps[agent_term_usage == t])]

        human_to_CCC += [evaluate.mean_rand_index(human_maps[human_term_usage == t],
                                                  ccc_maps[ccc_term_usage == t])]

        agent_to_CCC += [evaluate.mean_rand_index(agent_maps[agent_term_usage == t],
                                                  ccc_maps[ccc_term_usage == t])]

        human_to_random += [evaluate.mean_rand_index(human_maps[human_term_usage == t],
                                                     [[np.random.randint(t) for n in range(330)] for n in range(100)])]

    # print human vs machine
    exp.log('\n'.join(
        [
            'Terms & H-H & RL-RL & H-RL & H-CCC & RL-CCC & H-R \\\\ \\thickhline'] +
        ['{:2d} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}\\\\ \\hline'.format(
            term_usage_to_analyse[i],
            human_to_human[i],
            agent_to_agent[i],
            human_to_agent[i],
            human_to_CCC[i],
            agent_to_CCC[i],
            human_to_random[i])
            for i in range(len(term_usage_to_analyse))
        ]))

    # print perception noise influence table
    exp.log('\n'.join(
        ['Terms used & Mean rand index for all & Mean rand index within noise group \\\\ \\thickhline'] +
        ['{:2d} & {:.3f} & {:.3f} \\\\ \\hline'.format(
            term_usage_to_analyse[i],
            agent_to_agent[i],
            agent_to_agent_per_noise[i])
            for i in range(len(term_usage_to_analyse))
        ]))


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

    print_tables(exp)


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

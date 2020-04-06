import numpy as np
import viz
import matplotlib.pyplot as plt
import gridengine as sge
import math
import evaluate
from gridengine.pipeline import Experiment
import com_enviroments
import exp_shared

import exp_color_rl
import exp_color_ccc
import exp_color_random
import exp_color_human

from PIL import Image

import Correlation_Clustering


def run(host_name='local', pipeline='', exp_rl_id='', exp_ccc_id='', exp_random_id='', exp_human_id='', exp_old_id=''):
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
                                   ('exp_human_id', exp_human_id),
                                   ('exp_old_id', exp_old_id)])
    exp.save()

    return exp


def visualize(exp):

    cost_plot(exp, 'regier_cost', ylabel='KL Loss', xlabel='Color terms used', xlim=[3, 11], ledgend_anchor=[0.53, 0.8])
    cost_plot(exp, 'wellformedness', ylabel='Well-formedness', xlabel='Color terms used', ylim=[30000, 50000], xlim=[3, 11])


def cost_plot(exp, measure_id, ylabel='', xlabel='', ylim=None, xlim=None, ledgend_anchor=None):
    group_by_measure_id = 'term_usage'
    fig, ax = plt.subplots()

    add_line_to_axes(ax, exp_color_ccc.run(pipeline=exp.fixed_params['exp_ccc_id']), measure_id, group_by_measure_id, line_label='CIELAB correlation clustering')
    add_line_to_axes(ax, exp_color_random.run(pipeline=exp.fixed_params['exp_random_id']), measure_id, group_by_measure_id, line_label='Random')
    add_scatter_to_axes(ax, exp_color_human.run(pipeline=exp.fixed_params['exp_human_id']), measure_id, group_by_measure_id, label='WCS languages')
    add_line_to_axes(ax, exp_color_human.run(pipeline=exp.fixed_params['exp_human_id']), measure_id, group_by_measure_id, line_label='WCS languages mean', conf=False)
    add_line_to_axes(ax, exp_color_rl.run(pipeline=exp.fixed_params['exp_rl_id']), measure_id, group_by_measure_id, line_label='RL discrete')
    add_line_to_axes(ax, exp_color_rl.run(pipeline=exp.fixed_params['exp_old_id']), measure_id, group_by_measure_id, line_label='RL continous')
    if not ledgend_anchor is None:
        ax.legend(loc="upper left", bbox_to_anchor=(ledgend_anchor[0], ledgend_anchor[1]))
    else:
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

    plt.savefig(fig_name, dpi=300, compression="tiff_lzw")
    img = Image.open(fig_name)
    # (3) save as TIFF
    img.save(fig_name + '.tiff', dpi=(300, 300))
    img.close()


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
    term_usage_to_analyse = list(range(3, 11))

    exp_rl = exp_color_rl.run(pipeline=exp.fixed_params['exp_rl_id'])
    exp_old = exp_color_rl.run(pipeline=exp.fixed_params['exp_old_id'])

    agent_maps = exp_rl.reshape('agent_language_map')
    agent_term_usage = exp_rl.reshape('term_usage')
    maps_vs_noise = exp_rl.reshape('agent_language_map', as_function_of_axes=['perception_noise'])
    term_usage_vs_noise = exp_rl.reshape('term_usage', as_function_of_axes=['perception_noise'])

    # Old exp
    old_agent_maps = exp_old.reshape('agent_language_map')
    old_agent_term_usage = exp_old.reshape('term_usage')
    old_maps_vs_noise = exp_old.reshape('agent_language_map', as_function_of_axes=['perception_noise'])
    old_term_usage_vs_noise = exp_old.reshape('term_usage', as_function_of_axes=['perception_noise'])


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

    # t-test
    t_disc_cont = []
    t_disc_cont_human = []
    t_disc_cont_ccc = []

    discrete_to_continous = []
    old_to_old = []
    for t in term_usage_to_analyse:
        print('Nbr terms: {}'.format(t))
        print('Nbr exp discrete: {}'.format(agent_maps[agent_term_usage == t].shape[0]))
        print('Nbr exp continous: {}'.format(old_agent_maps[old_agent_term_usage == t].shape[0]))
        agent_to_agent += [evaluate.mean_rand_index(agent_maps[agent_term_usage == t])]
        old_to_old += [evaluate.mean_rand_index(old_agent_maps[old_agent_term_usage == t])]
        discrete_to_continous += [evaluate.mean_rand_index(agent_maps[agent_term_usage == t],
                                                  old_agent_maps[old_agent_term_usage == t])]
        mc = []
        for noise_i in range(len(maps_vs_noise)):
            tmp = maps_vs_noise[noise_i][term_usage_vs_noise[noise_i] == t]
            if tmp.shape[0] > 1:
                mc += [evaluate.mean_rand_index(tmp)]

        mc = np.array(mc)
        agent_to_agent_per_noise += [tuple(mc[:, i][~np.isnan(mc[:, i])].mean() for i in range(2))]
 #       a = np.array([evaluate.mean_rand_index(maps_vs_noise[noise_i][term_usage_vs_noise[noise_i] == t])
 #                     for noise_i in range(len(maps_vs_noise))])
 #       a_2= []
 #       for element in a:
 #           if isinstance(element, list):
 #               a_2 += element
 #           elif isinstance(element, tuple) or isinstance(element, np.ndarray):
 #               a_2 += [e for e in element]
 #           else:
 #               a_2 += [element]
 #       a = np.array(a_2)
 #       agent_to_agent_per_noise += [a[~np.isnan(a)].mean()]
        human_to_human += [evaluate.mean_rand_index(human_maps[human_term_usage == t])]
        human_to_agent += [evaluate.mean_rand_index(human_maps[human_term_usage == t],
                                                    agent_maps[agent_term_usage == t])]

        human_to_CCC += [evaluate.mean_rand_index(human_maps[human_term_usage == t],
                                                  ccc_maps[ccc_term_usage == t])]

        agent_to_CCC += [evaluate.mean_rand_index(agent_maps[agent_term_usage == t],
                                                  ccc_maps[ccc_term_usage == t])]

        human_to_random += [evaluate.mean_rand_index(human_maps[human_term_usage == t],
                                                     [[np.random.randint(t) for n in range(330)] for n in range(100)])]

        # t_test
        # cont vs disc
        t_disc_cont += [evaluate.check_hypothesis(ce_a=old_agent_maps[old_agent_term_usage == t], ce_b=agent_maps[agent_term_usage == t])]
        t_disc_cont_human += [evaluate.check_hypothesis(ce_a=old_agent_maps[old_agent_term_usage == t], ce_b=agent_maps[agent_term_usage == t], ce_c=human_maps[human_term_usage == t])]
        t_disc_cont_ccc += [evaluate.check_hypothesis(ce_a=old_agent_maps[old_agent_term_usage == t], ce_b=agent_maps[agent_term_usage == t], ce_c=ccc_maps[ccc_term_usage == t])]
    print(term_usage_to_analyse)
    # print human vs machine
    exp.log('\n'.join(
        [
            '\nTerms & H-H & discrete-discrete & H-discrete & H-CCC & discrete-CCC & H-R & continous-continous & discrete-continous \\\\ \\thickhline'] +
        ['{:2d} & {:.3f}($\pm${:.3f}) & {:.3f}($\pm${:.3f}) & {:.3f}($\pm${:.3f}) & {:.3f}($\pm${:.3f}) & {:.3f}($\pm${:.3f}) & {:.3f}($\pm${:.3f}) & {:.3f}($\pm${:.3f}) & {:.3f}($\pm${:.3f})\\\\ \\hline'.format(
            term_usage_to_analyse[i],
            human_to_human[i][0],human_to_human[i][1],
            agent_to_agent[i][0],agent_to_agent[i][1],
            human_to_agent[i][0],human_to_agent[i][1],
            human_to_CCC[i][0],human_to_CCC[i][1],
            agent_to_CCC[i][0],agent_to_CCC[i][1],
            human_to_random[i][0],human_to_random[i][1],
            old_to_old[i][0], old_to_old[i][1],
            discrete_to_continous[i][0], discrete_to_continous[i][1]
            )
            for i in range(len(term_usage_to_analyse))
        ]).replace('0.', '.'))


    # print stat-test
    exp.log('\n'.join(
        [
            '\nTerms & t-statistic & p-value\\\\ \\thickhline'] +
        ['{:2d} & {:.3f} & {:.3f}\\\\ \\hline'.format(
            term_usage_to_analyse[i],
            t_disc_cont[i][0],t_disc_cont[i][1],
            )
            for i in range(len(term_usage_to_analyse))
        ]).replace('0.', '.'))

    exp.log('\n'.join(
        [
            '\nTerms & t-statistic & p-value\\\\ \\thickhline'] +
        ['{:2d} & {:.3f} & {:.3f}\\\\ \\hline'.format(
            term_usage_to_analyse[i],
            t_disc_cont_human[i][0],t_disc_cont_human[i][1],
            )
            for i in range(len(term_usage_to_analyse))
        ]).replace('0.', '.'))

    exp.log('\n'.join(
        [
            '\nTerms & t-statistic & p-value\\\\ \\thickhline'] +
        ['{:2d} & {:.3f} & {:.3f}\\\\ \\hline'.format(
            term_usage_to_analyse[i],
            t_disc_cont_ccc[i][0],t_disc_cont_ccc[i][1],
            )
            for i in range(len(term_usage_to_analyse))
        ]).replace('0.', '.'))
    # print perception noise influence table
   # exp.log('\n'.join(
   #     ['\nTerms used & All & Within noise group \\\\ \\thickhline'] +
   #     ['{:2d} & {:.3f}($\pm${:.3f}) & {:.3f}($\pm${:.3f}) \\\\ \\hline'.format(
   #         term_usage_to_analyse[i],
   #         agent_to_agent[i][0],agent_to_agent[i][1],
   #         agent_to_agent_per_noise[i][0],agent_to_agent_per_noise[i][1])
   #         for i in range(len(term_usage_to_analyse))
   #     ]))


def main(args):
    # Run experiment
    exp = run(args.host_name,
              pipeline=args.pipeline,
              exp_rl_id=args.exp_rl_id,
              exp_ccc_id=args.exp_ccc_id,
              exp_random_id=args.exp_random_id,
              exp_human_id=args.exp_human_id,
              exp_old_id=args.exp_old_id)

    # Visualize experiment
    #visualize(exp)

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
    parser.add_argument('--exp_old_id', type=str, default='',
                        help='Compare to old RL-run')
    main(parser.parse_args())

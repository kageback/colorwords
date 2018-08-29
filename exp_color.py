import numpy as np
import gridengine as sge
import viz
import evaluate
from gridengine.pipeline import Experiment
import com_enviroments
import exp_shared

import exp_color_rl
import exp_color_cielab_cc
import Correlation_Clustering

def run(host_name, pipeline='', exp_rl_id='', exp_ccc_id=''):
    if pipeline != '':
        return exp_shared.load_exp(pipeline)

    exp = Experiment(exp_name='color_dev',
                     fixed_params=[('exp_rl_id', exp_rl_id),
                                   ('exp_ccc_id', exp_ccc_id)])

    # RL experiment
    exp_rl = exp_color_rl.run(host_name, pipeline=exp.fixed_params['exp_rl_id'])
    exp.set_result('exp_rl', value=exp_rl)

    # cielab correlation clustering experiment
    exp_ccc = exp_color_cielab_cc.run(host_name, pipeline=exp.fixed_params['exp_ccc_id'])
    exp.set_result('exp_ccc', value=exp_ccc)

    exp.save()

    return exp


def visualize(exp):
    exp_rl = exp.get('exp_rl')
    exp_ccc = exp.get('exp_ccc')

    term_usage_to_analyse = list(range(3, 12))
    iter = 10

    agent_maps = exp_rl.reshape('agent_language_map')
    agent_term_usage = exp_rl.reshape('term_usage')

    maps_vs_noise = exp_rl.reshape('agent_language_map', as_function_of_axes=['perception_noise'])
    term_usage_vs_noise = exp_rl.reshape('term_usage', as_function_of_axes=['perception_noise'])

    wcs = com_enviroments.make('wcs')
    human_maps = np.array(list(wcs.human_mode_maps.values()))
    human_term_usage = np.array([np.unique(m).shape[0] for m in human_maps])

    agent_mean_rand_vs_term_usage = []
    agent_mean_rand_over_noise_groups_vs_term_usage = []
    human_mean_rand_vs_term_usage = []
    cross_rand_vs_term_usage = []
    cross_agent_consensus_to_humans_vs_term_usage = []
    human_to_cielab_rand = []
    human_to_random_rand = []

    for t in term_usage_to_analyse:
        agent_mean_rand_vs_term_usage += [evaluate.mean_rand_index(agent_maps[agent_term_usage == t])]

        a = np.array([evaluate.mean_rand_index(maps_vs_noise[noise_i][term_usage_vs_noise[noise_i] == t])
          for noise_i in range(len(maps_vs_noise))])

        agent_mean_rand_over_noise_groups_vs_term_usage += [a[~np.isnan(a)].mean()]

        human_mean_rand_vs_term_usage += [evaluate.mean_rand_index(human_maps[human_term_usage == t])]

        cross_rand_vs_term_usage += [evaluate.mean_rand_index(human_maps[human_term_usage == t],
                                                              agent_maps[agent_term_usage == t])]
        if len(agent_maps[agent_term_usage == t]) >= 1:
            agent_consensus_map = Correlation_Clustering.compute_consensus_map(agent_maps[agent_term_usage == t], k=t, iter=iter)
            cross_agent_consensus_to_humans_vs_term_usage += [evaluate.mean_rand_index(human_maps[human_term_usage == t],
                                                                                       [agent_consensus_map])]
            wcs.plot_with_colors(agent_consensus_map,
                               save_to_path=exp_rl.pipeline_path + 'agent_consensus_map-' + str(t) + '_terms.png')
        else:
            cross_agent_consensus_to_humans_vs_term_usage += [np.nan]

        human_to_cielab_rand += [evaluate.mean_rand_index(human_maps[human_term_usage == t],
                                                          [evaluate.compute_cielab_map(wcs, k=t, iterations=10)])]

        human_to_random_rand += [evaluate.mean_rand_index(human_maps[human_term_usage == t], [[np.random.randint(t) for n in range(330)] for n in range(100)])]


    # print perception noise influence table
    exp_rl.log.info('\n'.join(
        ['Terms used & Mean rand index for all & Mean rand index within noise group \\\\ \\thickhline'] +
        ['{:2d} & {:.3f} & {:.3f} \\\\ \\hline'.format(
            term_usage_to_analyse[i],
            agent_mean_rand_vs_term_usage[i],
            agent_mean_rand_over_noise_groups_vs_term_usage[i])
            for i in range(len(term_usage_to_analyse))
        ]))

    #print human vs machine
    exp_rl.log.info('\n'.join(
        ['Terms used & Human mean rand index for all & Agents mean rand index & Cross human agent rand index & cross agent consensus to human \\\\ \\thickhline'] +
        ['{:2d} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}\\\\ \\hline'.format(
            term_usage_to_analyse[i],
            human_mean_rand_vs_term_usage[i],
            agent_mean_rand_vs_term_usage[i],
            cross_rand_vs_term_usage[i],
            cross_agent_consensus_to_humans_vs_term_usage[i],
            human_to_cielab_rand[i],
            human_to_random_rand[i])
            for i in range(len(term_usage_to_analyse))
        ]))

    # term usage across different level of noise
    viz.plot_with_conf(exp_rl, 'term_usage', 'perception_noise', 'com_noise',
                       x_label='perception $\sigma^2$',
                       z_label='com $\sigma^2$', )
    viz.hist(exp_rl, 'term_usage', 'perception_noise')




def com_plots(exp_rl):
    viz.plot_with_conf2(exp_rl, 'regier_cost', 'term_usage', 'com_noise', z_label='com $\sigma^2$')
    viz.plot_with_conf2(exp_rl, 'gibson_cost', 'term_usage', 'com_noise', z_label='com $\sigma^2$')
    viz.plot_with_conf2(exp_rl, 'wellformedness', 'term_usage', 'com_noise', z_label='com $\sigma^2$')


def main(args):
    # Run experiment
    exp = run(args.host_name, pipeline=args.pipeline, exp_rl_id=args.exp_rl_id, exp_ccc_id=args.exp_ccc_id)

    # Visualize experiment
    # visualize(exp)

    # Todo regier_cost/gibson_cost/wellformedness per term usage including baselines

    com_plots(exp.get('exp_rl'))


if __name__ == "__main__":
    parser = exp_shared.parse_script_arguments()
    parser.add_argument('--exp_ccc_id', type=str, default='',
                        help='cielab correlation clustering experiment')
    parser.add_argument('--exp_rl_id', type=str, default='',
                        help='Reinforcement learning experiment')
    main(parser.parse_args())

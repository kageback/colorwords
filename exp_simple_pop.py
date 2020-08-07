import numpy as np

import Correlation_Clustering
import gridengine as sge
import com_game
import viz
import evaluate
from gridengine.pipeline import Experiment
import com_enviroments
from agents import SoftmaxAgent
import exp_shared
import matplotlib.pyplot as plt
from graph import PopulationGraph
from population_game import PopulationGame

def run(host_name='local', pipeline=''):
    if pipeline != '':
        return exp_shared.load_exp(pipeline)

    # Create and run new experiment
    queue = exp_shared.create_queue(host_name)
    queue.sync('.', '.', exclude=['pipelines/*', 'fig/*', 'old/*', 'cogsci/*'], sync_to=sge.SyncTo.REMOTE,
               recursive=True)
    exp = Experiment(exp_name='rl_population',
                     fixed_params=[('loss_type', 'REINFORCE'),
                                   ('bw_boost', 1),
                                   ('env', 'wcs'),
                                   ('max_epochs', 300),  # 10000
                                   ('hidden_dim', 20),
                                   ('batch_size', 100),
                                   ('perception_dim', 3),
                                   ('target_dim', 330),
                                   ('print_interval', 1000),
                                   ('msg_dim', 50)],
                     param_ranges=[('avg_over', range(2)),
                                   ('perception_noise',[0]),
                                   ('com_noise', [0]),
                                   ('n_agents', [2, 4])],
                     queue=queue)
    queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.REMOTE, recursive=True)

    env = exp.run(com_enviroments.make, exp.fixed_params['env']).result()
    exp_i = 0
    for (params_i, params_v) in exp:
        print('Scheduled %d experiments out of %d' % (exp_i, len(list(exp))))
        exp_i += 1
        agent_list = []
        for _ in range(params_v[exp.axes['n_agents']]):
            agent_list.append(SoftmaxAgent(msg_dim=exp.fixed_params['msg_dim'],
                                      hidden_dim=exp.fixed_params['hidden_dim'],
                                      color_dim=exp.fixed_params['target_dim'],
                                      perception_dim=exp.fixed_params['perception_dim']))


        graph = PopulationGraph(agents=agent_list)
        game = PopulationGame(max_epochs=exp.fixed_params['max_epochs'],
                                        perception_noise=params_v[exp.axes['perception_noise']],
                                        batch_size=exp.fixed_params['batch_size'],
                                        print_interval=exp.fixed_params['print_interval'],
                                        perception_dim=exp.fixed_params['perception_dim'])

        agent_a = exp.run(game.play, graph, env).result()
        exp.set_result('agent_a', params_i, agent_a)



    return exp


def analyse(exp):
    env = exp.run(com_enviroments.make, exp.fixed_params['env']).result()
    i = 0
    for (params_i, params_v) in exp:
        agent_a = exp.get_result('agent_a', params_i)
        V = exp.run(evaluate.agent_language_map, env, a=agent_a).result()
        exp.set_result('agent_language_map', params_i, V)
        exp.set_result('gibson_cost', params_i, exp.run(evaluate.compute_gibson_cost2, env, a=agent_a).result())
        exp.set_result('regier_cost', params_i, exp.run(evaluate.regier2, env, map=V).result())
        exp.set_result('wellformedness', params_i, exp.run(evaluate.wellformedness, env, V=V).result())
        exp.set_result('term_usage', params_i, exp.run(evaluate.compute_term_usage, V=V).result())

        print('Scheduled analysis of %d experiments out of %d' % (i, len(list(exp))))
        i += 1

def visualize(exp):
    print('Visualize results')

    #viz.plot_with_conf(exp, 'term_usage', 'com_noise', x_label='communication $\sigma^2$')
    viz.plot_with_conf(exp, 'term_usage', 'n_agents', x_label='Population size')
    viz.plot_with_conf(exp, 'gibson_cost', 'n_agents', x_label='Population size')
    viz.plot_with_conf(exp, 'regier_cost', 'n_agents', x_label='Population size')

    viz.plot_with_conf2(exp,
                         'gibson_cost', 'term_usage', 'n_agents',
                         measure_label='Expected surprise',
                         group_by_measure_label='Color terms used',
                         z_label='Population size')


    viz.plot_with_conf2(exp,
                         'wellformedness', 'term_usage', 'n_agents',
                         measure_label='Wellformedness',
                         group_by_measure_label='Color terms used',
                         z_label='Population size')

    viz.plot_with_conf2(exp,
                         'regier_cost', 'term_usage', 'n_agents',
                         measure_label='KL loss',
                         group_by_measure_label='Color terms used',
                         z_label='Population size')

    viz.plot_with_conf(exp, 'term_usage', 'perception_noise', x_label='environment $\sigma^2$')

    # term usage across different level of noise
    # viz.plot_with_conf(exp, 'term_usage', 'com_noise', x_label='com $\sigma^2$')
    # viz.plot_lines_with_conf(exp, 'term_usage', 'com_noise', 'perception_noise', x_label='com $\sigma^2$',
    #                              z_label='perception $\sigma^2$')

    #    viz.plot_lines_with_conf(exp, 'term_usage', 'perception_noise', 'com_noise', x_label='perception $\sigma^2$',
    #                         z_label='com $\sigma^2$', )

    # viz.plot_with_conf2(exp, 'regier_cost', 'term_usage', 'com_noise', measure_label='KL Loss', z_label='com $\sigma^2$')
    # viz.plot_with_conf2(exp, 'wellformedness', 'term_usage', 'com_noise', measure_label='Well-formedness', z_label='com $\sigma^2$')


def print_tables(exp):
    term_usage_to_analyse = list(range(3, 12))
    iter = 10

    agent_maps = exp.reshape('agent_language_map')
    agent_term_usage = exp.reshape('term_usage')
    maps_vs_noise = exp.reshape('agent_language_map', as_function_of_axes=['perception_noise'])
    term_usage_vs_noise = exp.reshape('term_usage', as_function_of_axes=['perception_noise'])

    e = com_enviroments.make('wcs')
    human_maps = np.array(list(e.human_mode_maps.values()))
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
            agent_consensus_map = Correlation_Clustering.compute_consensus_map(agent_maps[agent_term_usage == t], k=t,
                                                                               iter=iter)
            cross_agent_consensus_to_humans_vs_term_usage += [
                evaluate.mean_rand_index(human_maps[human_term_usage == t],
                                         [agent_consensus_map])]
            e.plot_with_colors(agent_consensus_map,
                               save_to_path=exp.pipeline_path + 'agent_consensus_map-' + str(t) + '_terms.png')
        else:
            cross_agent_consensus_to_humans_vs_term_usage += [np.nan]

        human_to_cielab_rand += [evaluate.mean_rand_index(human_maps[human_term_usage == t],
                                                          [evaluate.compute_cielab_map(e, k=t, iterations=10)])]

        human_to_random_rand += [evaluate.mean_rand_index(human_maps[human_term_usage == t],
                                                          [[np.random.randint(t) for n in range(330)] for n in
                                                           range(100)])]



def main(args):

    # Run experiment
    exp = run(args.host_name, pipeline=args.pipeline)

    # evaluate results
    if args.pipeline == '':
        analyse(exp)

        print("\nAll tasks queued to clusters")
        exp.save()

        # wait for all tasks to complete and sync back result
        exp.wait(retry_interval=5)
        exp.queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.LOCAL, recursive=True)


    # Visualize experiment
    #visualize(exp)

    #print_tables(exp)


if __name__ == "__main__":
    main(exp_shared.parse_script_arguments().parse_args())

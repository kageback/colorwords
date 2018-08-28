import argparse
import numpy as np

import Correlation_Clustering
import gridengine as sge
import com_game
import viz
import evaluate
from gridengine.pipeline import Experiment
import com_enviroments
import agents
import exp_shared

def run(host_name):
    # Create and run new experiment
    queue = exp_shared.create_queue(host_name)
    queue.sync('.', '.', exclude=['pipelines/*', 'fig/*', 'old/*', 'cogsci/*'], sync_to=sge.SyncTo.REMOTE,
               recursive=True)
    exp = Experiment(exp_name='num_b',
                     fixed_params=[('env', 'numbers'),
                                   ('max_epochs', 10000),  #10000
                                   ('hidden_dim', 10),
                                   ('batch_size', 100),
                                   ('perception_dim', 1),
                                   ('target_dim', 100),
                                   ('print_interval', 1000)],
                     param_ranges=[('avg_over', range(20)),  # 50
                                   ('perception_noise', [0]),  # [0, 25, 50, 100],
                                   ('msg_dim', range(1, 10)), #3, 12
                                   ('com_noise', np.linspace(start=0, stop=1, num=11))
                                   ],
                     queue=queue)
    queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.REMOTE, recursive=True)

    env = exp.run(com_enviroments.make, exp.fixed_params['env']).result()
    exp_i = 0
    for (params_i, params_v) in exp:
        print('Scheduled %d experiments out of %d' % (exp_i, len(list(exp))))
        exp_i += 1
        #print('Param epoch %d of %d' % (params_i[exp.axes['avg_over']], exp.shape[exp.axes['avg_over']]))

        agent_a = agents.SoftmaxAgent(msg_dim=params_v[exp.axes['msg_dim']],
                                      hidden_dim=exp.fixed_params['hidden_dim'],
                                      color_dim=exp.fixed_params['target_dim'],
                                      perception_dim=exp.fixed_params['perception_dim'])

        agent_b = agents.SoftmaxAgent(msg_dim=params_v[exp.axes['msg_dim']],
                                      hidden_dim=exp.fixed_params['hidden_dim'],
                                      color_dim=exp.fixed_params['target_dim'],
                                      perception_dim=exp.fixed_params['perception_dim'])

        game = com_game.NoisyChannelGame(reward_func='abs_dist',
                                         com_noise=params_v[exp.axes['com_noise']],
                                         msg_dim=params_v[exp.axes['msg_dim']],
                                         max_epochs=exp.fixed_params['max_epochs'],
                                         perception_noise=params_v[exp.axes['perception_noise']],
                                         batch_size=exp.fixed_params['batch_size'],
                                         print_interval=exp.fixed_params['print_interval'],
                                         perception_dim=exp.fixed_params['perception_dim'],
                                         loss_type='CrossEntropyLoss')

        game_outcome = exp.run(game.play, env, agent_a, agent_b).result()

        V = exp.run(game.agent_language_map, env, a=game_outcome).result()

        exp.set_result('agent_language_map', params_i, V)
        exp.set_result('gibson_cost', params_i, exp.run(game.compute_gibson_cost, env, a=game_outcome).result(1))
        exp.set_result('regier_cost', params_i, exp.run(game.communication_cost_regier, env, V=V).result())
        exp.set_result('wellformedness', params_i, exp.run(game.wellformedness, env, V=V).result())
        exp.set_result('term_usage', params_i, exp.run(game.compute_term_usage, V=V).result())


    print("\nAll tasks queued to clusters")

    # wait for all tasks to complete
    exp.save()
    exp.wait(retry_interval=5)
    queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.LOCAL, recursive=True)

    return exp

def visualize(exp):
    print('plot results')

    V_mode = com_game.BaseGame.reduce_maps('agent_language_map', exp, reduce_method='mode')
    ranges = com_game.BaseGame.compute_ranges(V_mode)
    print(ranges)

    # gibson cost
    viz.plot_with_conf(exp, 'gibson_cost', 'msg_dim', 'perception_noise', measure_label='Gibson communication efficiency', x_label='number of color words', z_label='perception $\sigma^2$')
    viz.plot_with_conf(exp, 'gibson_cost', 'msg_dim', 'com_noise', measure_label='Gibson communication efficiency', x_label='number of color words', z_label='com $\sigma^2$')
    # viz.plot_with_conf(exp, 'gibson_cost', 'com_noise', 'perception_noise', measure_label='Gibson communication efficiency')

    # regier cost
    viz.plot_with_conf(exp, 'regier_cost', 'msg_dim', 'perception_noise', x_label='number of color words', z_label='perception $\sigma^2$')
    viz.plot_with_conf(exp, 'regier_cost', 'msg_dim', 'com_noise', x_label='number of color words', z_label='com $\sigma^2$')

    # wellformedness
    viz.plot_with_conf(exp, 'wellformedness', 'msg_dim', 'perception_noise', x_label='number of color words', z_label='perception $\sigma^2$')
    viz.plot_with_conf(exp, 'wellformedness', 'msg_dim', 'com_noise', x_label='number of color words', z_label='com $\sigma^2$')

    # term usage
    viz.plot_with_conf(exp, 'term_usage', 'msg_dim', 'perception_noise', x_label='number of color words', z_label='perception $\sigma^2$')
    viz.plot_with_conf(exp, 'term_usage', 'msg_dim', 'com_noise', x_label='number of color words', z_label='com $\sigma^2$')


def main():
    args = exp_shared.parse_script_arguments()
    # Run experiment
    if args.pipeline == '':
        exp = run(args.host_name)
    else:
        # Load existing experiment
        exp = Experiment.load(args.pipeline)
        if args.resync == 'y':
            exp.wait(retry_interval=5)
            exp.queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.LOCAL, recursive=True)


    cluster_ensemble = exp.get_flattened_results('agent_language_map')
    consensus = Correlation_Clustering.compute_consensus_map(cluster_ensemble, k=10, iter=100)
    print(consensus.values())

    # Visualize experiment
    visualize(exp)


if __name__ == "__main__":
    main()


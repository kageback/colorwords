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
import number_utils

def run(host_name='local', pipeline=''):
    if pipeline != '':
        return exp_shared.load_exp(pipeline)
    # Create and run new experiment
    queue = exp_shared.create_queue(host_name)
    queue.sync('.', '.', exclude=['pipelines/*', 'fig/*', 'old/*', 'cogsci/*'], sync_to=sge.SyncTo.REMOTE,
               recursive=True)
    exp = Experiment(exp_name='num_b',
                     fixed_params=[('env', 'numbers'),
                                   ('max_epochs', 10000),  #25000
                                   ('hidden_dim', 25),
                                   ('batch_size', 100),
                                   ('perception_dim', 1),
                                   ('target_dim', 100),
                                   ('print_interval', 1000)],
                     param_ranges=[('avg_over', range(1)),  # 50
                                   ('perception_noise', [0]),  # [0, 25, 50, 100],
                                   ('msg_dim', [50]), #3, 12
                                   ('sender_entropy', [0]),
                                   ('reciever_entropy', [0]),
                                   ('learning_rate', [0.0001]),
                                   ('com_noise', np.linspace(start=1, stop=1, num=1))
                                   ],
                     queue=queue)
    queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.REMOTE, recursive=True)

    env = exp.run(com_enviroments.make, exp.fixed_params['env'], data_dim=exp.fixed_params['target_dim']).result()
    exp_i = 0
    for (params_i, params_v) in exp:
        print('Scheduled %d experiments out of %d' % (exp_i, len(list(exp))))
        exp_i += 1
        #print('Param epoch %d of %d' % (params_i[exp.axes['avg_over']], exp.shape[exp.axes['avg_over']]))

        agent_a = agents.SoftmaxAgent(msg_dim=params_v[exp.axes['msg_dim']],
                                      hidden_dim=exp.fixed_params['hidden_dim'],
                                      color_dim=exp.fixed_params['target_dim'],
                                      perception_dim=exp.fixed_params['perception_dim'])

        agent_b = agents.GaussianAgent(msg_dim=params_v[exp.axes['msg_dim']],
                                      hidden_dim=exp.fixed_params['hidden_dim'],
                                      color_dim=exp.fixed_params['target_dim'],
                                      perception_dim=exp.fixed_params['perception_dim'])

        game = com_game.GaussianGame(reward_func='abs_dist',
                                         com_noise=params_v[exp.axes['com_noise']],
                                         msg_dim=params_v[exp.axes['msg_dim']],
                                         max_epochs=exp.fixed_params['max_epochs'],
                                         perception_noise=params_v[exp.axes['perception_noise']],
                                         batch_size=exp.fixed_params['batch_size'],
                                         print_interval=exp.fixed_params['print_interval'],
                                         perception_dim=exp.fixed_params['perception_dim'],
                                         sender_entropy=params_v[exp.axes['sender_entropy']],
                                         reciever_entropy=params_v[exp.axes['reciever_entropy']],
                                         learning_rate=params_v[exp.axes['learning_rate']],
                                         loss_type='REINFORCE')

        game_outcome = exp.run(game.play, env, agent_a, agent_b)
        sender = game_outcome.result(0)
        reciever = game_outcome.result(1)
        V = exp.run(evaluate.agent_language_map, env, a=sender).result()
        exp.set_result('agent_language_map', params_i, V)
        exp.set_result('gibson_cost', params_i, exp.run(game.compute_gibson_cost, env, a=sender).result(1))
        exp.set_result('regier_cost', params_i, exp.run(evaluate.communication_cost_regier, env, V=V).result())
        exp.set_result('wellformedness', params_i, exp.run(evaluate.wellformedness, env, V=V).result())
        exp.set_result('term_usage', params_i, exp.run(evaluate.compute_term_usage, V=V).result())
        # New metrics
      #  exp.set_result('reciever_prob', params_i, exp.run(number_utils.reciever_probs, V=V, reciever=reciever).result(0))
      #  exp.set_result('reciever_guess', params_i, exp.run(number_utils.reciever_probs, V=V, reciever=reciever).result(1))
        exp.set_result('sender_fraction_response', params_i, exp.run(number_utils.fraction_response, env, sender=sender).result())

    print("\nAll tasks queued to clusters")

    # wait for all tasks to complete
    exp.save()
    exp.wait(retry_interval=5)
    queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.LOCAL, recursive=True)

    return exp

def analyze(exp):
    # compute fraction of response
    fractions = number_utils.sort_fractions(exp)
    term_usage_to_analyse = list(range(2, 12))
    iter = 10
    agent_maps = exp.reshape('agent_language_map')
    agent_term_usage = exp.reshape('term_usage')
    agent_guess = exp.reshape('reciever_guess')
    print(agent_guess)
    for t in term_usage_to_analyse:

        if len(agent_maps[agent_term_usage == t]) >= 1:
            agent_consensus_map = number_utils.compute_consensus_map(agent_maps[agent_term_usage == t], k=t,
                                                                               iter=iter)
            #number_utils.plot_map(agent_consensus_map,
            #                   save_to_path=exp.pipeline_path + 'agent_consensus_map-' + str(t) + '_terms.png')
        # else:
            # compare to random and human partition


def visualize(exp):
    print('plot results')

    V_mode = com_game.BaseGame.reduce_maps('agent_language_map', exp, reduce_method='mode')
    ranges = com_game.BaseGame.compute_ranges(V_mode)
    print(ranges)
    # gibson cost
    viz.plot_lines_with_conf(exp, 'gibson_cost', 'msg_dim', 'perception_noise', measure_label='Gibson communication efficiency', x_label='number of words', z_label='perception $\sigma^2$')
    viz.plot_lines_with_conf(exp, 'gibson_cost', 'msg_dim', 'com_noise', measure_label='Gibson communication efficiency', x_label='number of words', z_label='com $\sigma^2$')
    # viz.plot_with_conf(exp, 'gibson_cost', 'com_noise', 'perception_noise', measure_label='Gibson communication efficiency')

    # regier cost
    viz.plot_lines_with_conf(exp, 'regier_cost', 'msg_dim', 'perception_noise', x_label='number of words', z_label='perception $\sigma^2$')
    viz.plot_lines_with_conf(exp, 'regier_cost', 'msg_dim', 'com_noise', x_label='number of words', z_label='com $\sigma^2$')

    # wellformedness
    viz.plot_lines_with_conf(exp, 'wellformedness', 'msg_dim', 'perception_noise', x_label='number of words', z_label='perception $\sigma^2$')
    viz.plot_lines_with_conf(exp, 'wellformedness', 'msg_dim', 'com_noise', x_label='number of  words', z_label='com $\sigma^2$')

    # term usage
    viz.plot_lines_with_conf(exp, 'term_usage', 'msg_dim', 'perception_noise', x_label='number of  words', z_label='perception $\sigma^2$')
    viz.plot_lines_with_conf(exp, 'term_usage', 'msg_dim', 'com_noise', x_label='number of words', z_label='com $\sigma^2$')


def main():
    args = exp_shared.parse_script_arguments().parse_args()
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

    #analyze(exp)
    # Visualize experiment
    visualize(exp)


if __name__ == "__main__":
    main()

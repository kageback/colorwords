import argparse
import numpy as np
import gridengine as sge
import com_game
import viz
from gridengine.pipeline import Experiment
from gridengine.queue import Queue, Local

import com_enviroments
import agents

def run():


    queue = Local()
    #queue = Queue(cluster_wd='~/runtime/colorwords/', host='titan.kageback.se', ge_gpu=1, queue_limit=4)
    #queue = Queue(cluster_wd='~/runtime/colorwords/', host='home.kageback.se', queue_limit=4)
    #queue = Queue(cluster_wd='~/runtime/colorwords/', host='ttitania.ce.chalmers.se', user='mlusers', queue_limit=4)

    queue.sync('.', '.', exclude=['pipelines/*', 'fig/*', 'old/*', 'cogsci/*'], sync_to=sge.SyncTo.REMOTE,
               recursive=True)

    exp = Experiment(exp_name='num_dev',
                     fixed_params=[('env', 'numbers'),
                                   ('max_epochs', 100),  #10000
                                   ('hidden_dim', 20),
                                   ('batch_size', 1000),
                                   ('perception_dim', 1),
                                   ('target_dim', 100),
                                   ('print_interval', 1000)],
                     param_ranges=[('avg_over', range(1)),  # 50
                                   ('perception_noise', [0]),  # [0, 25, 50, 100],
                                   ('msg_dim', range(10, 12)), #3, 12
                                   ('com_noise', [0.5,1])#np.linspace(start=0, stop=0.5, num=5))
                                   ],
                     queue=queue)
    queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.REMOTE, recursive=True)

    env = exp.run(com_enviroments.make, exp.fixed_params['env']).result()
    exp_i = 0
    for (params_i, params_v) in exp:
        print('Scheduled %d experiments out of %d' % (exp_i, len(list(exp))))
        exp_i += 1
        #print('Param epoch %d of %d' % (params_i[exp.axes['avg_over']], exp.shape[exp.axes['avg_over']]))

        agent_a = agent_b = agents.SoftmaxAgent(msg_dim=params_v[exp.axes['msg_dim']],
                                                hidden_dim=exp.fixed_params['hidden_dim'],
                                                color_dim=exp.fixed_params['target_dim'],
                                                perception_dim=exp.fixed_params['perception_dim'])

        game = com_game.NoisyChannelGame(reward_func='RMS_reward',
                                         com_noise=params_v[exp.axes['com_noise']],
                                         msg_dim=params_v[exp.axes['msg_dim']],
                                         max_epochs=exp.fixed_params['max_epochs'],
                                         perception_noise=params_v[exp.axes['perception_noise']],
                                         batch_size=exp.fixed_params['batch_size'],
                                         print_interval=exp.fixed_params['print_interval'],
                                         perception_dim=exp.fixed_params['perception_dim'])

        game_outcome = exp.run(game.play, env, agent_a, agent_b).result()

        V = exp.run(game.agent_language_map, env, a=game_outcome).result()

        #V = exp.run(env, call_member='agent_language_map', a=game_outcome)

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

    return exp.pipeline_name

def visualize(pipeline_name):
    print('plot results')
    exp = Experiment.load(pipeline_name)

    V_mode = com_game.BaseGame.reduce_maps('agent_language_map', exp, reduce_method='mode')
    ranges = com_game.BaseGame.compute_ranges(V_mode)
    print(ranges)
    # exp.run(env, call_member='print_ranges', V=V.result())
    viz.plot_result(exp, 'gibson_cost', 'com_noise', 'msg_dim',
                    measure_label='Gibson communication efficiency',
                    x_label='Communication noise',
                    z_label='terms')
    viz.plot_result(exp, 'regier_cost', 'com_noise', 'msg_dim')
    viz.plot_result(exp, 'wellformedness', 'com_noise', 'msg_dim')
    viz.plot_result(exp, 'term_usage', 'com_noise', 'msg_dim')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Numbers experiment')
    parser.add_argument('--pipeline', type=str, default='',
                        help='Name of existing pipeline to load for re-visualization')
    args = parser.parse_args()

    if args.pipeline == '':
        pipeline_name = run()
    visualize(pipeline_name)
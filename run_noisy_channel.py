import numpy as np

import evaluate
import gridengine as sge
import com_game
import viz
from gridengine.pipeline import Experiment
from gridengine.queue import Queue, Local

import com_enviroments
import agents

def main():


    queue = Local()
    #queue = Queue(cluster_wd='~/runtime/colorwords/', host='titan.kageback.se', ge_gpu=1, queue_limit=4)
    #queue = Queue(cluster_wd='~/runtime/colorwords/', host='home.kageback.se', queue_limit=4)
    #queue = Queue(cluster_wd='~/runtime/colorwords/', host='ttitania.ce.chalmers.se', user='mlusers', queue_limit=4)

    queue.sync('.', '.', exclude=['pipelines/*', 'fig/*', 'old/*', 'cogsci/*'], sync_to=sge.SyncTo.REMOTE,
               recursive=True)

    exp = Experiment(exp_name='noisy_channel',
                     fixed_params=[('env', 'wcs'),
                                   ('perception_noise', 0),
                                   ('max_epochs', 10000),  #10000
                                   ('hidden_dim', 20),
                                   ('batch_size', 100)],
                     param_ranges=[('avg_over', range(1)),
                                   ('msg_dim', range(6, 7)),  # 50
                                   ('com_noise', np.linspace(start=0, stop=1.5, num=2))],  # np.logspace(-2, 2, 10)
                     queue=queue)

    env = com_enviroments.make(exp.fixed_params['env'])


    for (params_i, params_v) in exp:
        print('Param epoch %d of %d' % (params_i[exp.axes['avg_over']], exp.shape[exp.axes['avg_over']]))

        agent_a = agent_b = agents.SoftmaxAgent(msg_dim=params_v[exp.axes['msg_dim']],
                                                hidden_dim=exp.fixed_params['hidden_dim'],
                                                color_dim=env.color_dim(),
                                                perception_dim=3)

        game = com_game.NoisyChannelContRewardGame(com_noise=params_v[exp.axes['com_noise']],
                                                   msg_dim=params_v[exp.axes['msg_dim']],
                                                   max_epochs=exp.fixed_params['max_epochs'],
                                                   perception_noise=exp.fixed_params['perception_noise'],
                                                   batch_size=exp.fixed_params['batch_size'],
                                                   print_interval=100)

        game_outcome = exp.run(game.play, env, agent_a, agent_b)

        V = exp.run(evaluate.color_graph_V, a=game_outcome.result(), env=env)

        exp.run(env.plot_with_colors, V=V.result(), save_to_path='test.png')

        exp.set_result('gibson_cost', params_i, exp.run(evaluate.compute_gibson_cost, a=game_outcome.result(), wcs=env))
        #exp.set_result('regier_cost', params_i, exp.run(wcs.communication_cost_regier, V=V.result()))
        #exp.set_result('wellformedness', params_i, exp.run(wcs.wellformedness, V=V.result()))
        #exp.set_result('combined_criterion', params_i, exp.run(wcs.combined_criterion, V=V.result()))
        #exp.set_result('term_usage', params_i, exp.run(wcs.compute_term_usage, V=V.result()))


    print("\nAll tasks queued to clusters")

    # wait for all tasks to complete
    exp.save()
    exp.wait(retry_interval=5)
    queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.LOCAL, recursive=True)

    print('plot results')
    viz.plot_com_noise_cost(exp)
    #viz.plot_reiger_gibson(exp)

    #viz.plot_wellformedness(exp)
    #viz.plot_combined_criterion(exp)
    #viz.plot_term_usage(exp)


if __name__ == "__main__":
    main()

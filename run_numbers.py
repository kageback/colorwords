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

    exp = Experiment(exp_name='num_dev',
                     fixed_params=[('env', 'numbers'),
                                   ('max_epochs', 10000),  # 10000
                                   ('hidden_dim', 20),
                                   ('batch_size', 100),
                                   ('sender_loss_multiplier', 100)],
                     param_ranges=[('avg_over', range(1)),
                                   ('msg_dim', range(3, 4)),  # 50
                                   ('com_noise', np.linspace(start=0, stop=1.5, num=1))],  # np.logspace(-2, 2, 10)
                     queue=queue)

    env = com_enviroments.make(exp.fixed_params['env'])

    for (params_i, params_v) in exp:
        print('Param epoch %d of %d' % (params_i[exp.axes['avg_over']], exp.shape[exp.axes['avg_over']]))

        agent_a = agent_b = agents.BasicAgent(msg_dim=params_v[exp.axes['msg_dim']],
                                                hidden_dim=exp.fixed_params['hidden_dim'],
                                                color_dim=env.color_dim(),
                                                perception_dim=3)

        net = exp.run(com_game.play_noisy_channel_desc_reward,
                      env=env,
                      agent_a=agent_a,
                      agent_b=agent_b,
                      com_noise=params_v[exp.axes['com_noise']],
                      msg_dim=params_v[exp.axes['msg_dim']],
                      max_epochs=exp.fixed_params['max_epochs'],
                      batch_size=exp.fixed_params['batch_size'],
                      print_interval=1000)

        V = exp.run(com_game.color_graph_V, a=net.result(), env=env)

        exp.set_result('gibson_cost', params_i, exp.run(evaluate.compute_gibson_cost, a=net.result(), wcs=env))


        exp.set_result('gibson_cost', params_i, exp.run(evaluate.compute_gibson_cost, a=net.result()))


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

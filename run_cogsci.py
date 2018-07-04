import evaluate
import gridengine as sge
import com_game
import viz
from gridengine.pipeline import Experiment
from gridengine.queue import Queue, Local

import com_enviroments
import agents

def main():


    #queue = Local()
    #queue = Queue(cluster_wd='~/runtime/colorwords/', host='titan.kageback.se', ge_gpu=1, queue_limit=4)
    #queue = Queue(cluster_wd='~/runtime/colorwords/', host='home.kageback.se', queue_limit=4)
    queue = Queue(cluster_wd='~/runtime/colorwords/', host='ttitania.ce.chalmers.se', user='mlusers', queue_limit=4)

    queue.sync('.', '.', exclude=['pipelines/*', 'fig/*', 'old/*', 'cogsci/*'], sync_to=sge.SyncTo.REMOTE,
               recursive=True)

    exp = Experiment(exp_name='cogsci',
                     fixed_params=[('env', 'wcs'),
                                   ('max_epochs', 10000),  #10000
                                   ('hidden_dim', 20),
                                   ('batch_size', 100),
                                   ('perception_dim', 3),
                                   ('target_dim', 330),
                                   ('sender_loss_multiplier', 100),
                                   ('print_interval', 1000)],
                     param_ranges=[('avg_over', range(2)),  # 50
                                   ('perception_noise', [0, 25]),  # [0, 25, 50, 100]
                                   ('msg_dim', range(3, 12))],  # range(3,12)
                     queue=queue)
    queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.REMOTE, recursive=True)

    env = exp.run(com_enviroments.make, exp.fixed_params['env'])

    for (params_i, params_v) in exp:
        print('Param epoch %d of %d' % (params_i[exp.axes['avg_over']], exp.shape[exp.axes['avg_over']]))

        agent_a = agent_b = agents.BasicAgent(msg_dim=params_v[exp.axes['msg_dim']],
                                              hidden_dim=exp.fixed_params['hidden_dim'],
                                              color_dim=exp.fixed_params['target_dim'],
                                              perception_dim=exp.fixed_params['perception_dim'])

        game = com_game.OneHotChannelContRewardGame(reward_func='regier_reward',
                                                    sender_loss_multiplier=exp.fixed_params['sender_loss_multiplier'],
                                                    msg_dim=params_v[exp.axes['msg_dim']],
                                                    max_epochs=exp.fixed_params['max_epochs'],
                                                    perception_noise=params_v[exp.axes['perception_noise']],
                                                    batch_size=exp.fixed_params['batch_size'],
                                                    print_interval=exp.fixed_params['print_interval'],
                                                    perception_dim=exp.fixed_params['perception_dim'])


        game_outcome = exp.run(game.play, env.result(), agent_a, agent_b)

        V = exp.run(env, call_member='agent_language_map', a=game_outcome.result())

        exp.run(env, call_member='plot_with_colors', V=V.result(), save_to_path=exp.pipeline_path + 'language_map.png')

        exp.set_result('gibson_cost', params_i, exp.run(env, call_member='compute_gibson_cost', a=game_outcome.result()))
        exp.set_result('regier_cost', params_i, exp.run(env, call_member='communication_cost_regier', V=V.result()))
        exp.set_result('wellformedness', params_i, exp.run(env, call_member='wellformedness', V=V.result()))
        exp.set_result('term_usage', params_i, exp.run(env, call_member='compute_term_usage', V=V.result()))


    print("\nAll tasks queued to clusters")

    # wait for all tasks to complete
    exp.save()
    exp.wait(retry_interval=5)
    queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.LOCAL, recursive=True)

    print('plot results')
    viz.plot_reiger_gibson(exp)
    viz.plot_wellformedness(exp)
    viz.plot_term_usage(exp)


if __name__ == "__main__":
    main()

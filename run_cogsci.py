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
                                   ('max_epochs', 100),  #10000
                                   ('hidden_dim', 20),
                                   ('batch_size', 100),
                                   ('perception_dim', 3),
                                   ('print_interval', 1000)],
                     param_ranges=[('avg_over', range(1)),  # 50
                                   ('perception_noise', [0]),  # [0, 25, 50, 100]
                                   ('msg_dim', range(3, 8))],  # range(3,12)
                     queue=queue)

    env = com_enviroments.make(exp.fixed_params['env'])

    for (params_i, params_v) in exp:
        print('Param epoch %d of %d' % (params_i[exp.axes['avg_over']], exp.shape[exp.axes['avg_over']]))

        agent_a = agent_b = agents.BasicAgent(msg_dim=params_v[exp.axes['msg_dim']],
                                              hidden_dim=exp.fixed_params['hidden_dim'],
                                              color_dim=env.data_dim(),
                                              perception_dim=exp.fixed_params['perception_dim'])

        game = com_game.OneHotChannelContRewardGame(reward_func='regier_reward',
                                                    sender_loss_multiplier=100,
                                                    msg_dim=params_v[exp.axes['msg_dim']],
                                                    max_epochs=exp.fixed_params['max_epochs'],
                                                    perception_noise=params_v[exp.axes['perception_noise']],
                                                    batch_size=exp.fixed_params['batch_size'],
                                                    print_interval=exp.fixed_params['print_interval'],
                                                    perception_dim=exp.fixed_params['perception_dim'])


        game_outcome = exp.run(game.play, env, agent_a, agent_b)

        V = exp.run(env.agent_language_map, a=game_outcome.result())

        exp.set_result('gibson_cost', params_i, exp.run(env.compute_gibson_cost, a=game_outcome.result()))
        exp.set_result('regier_cost', params_i, exp.run(env.communication_cost_regier, V=V.result(), sim=env.sim_np))
        exp.set_result('wellformedness', params_i, exp.run(env.wellformedness, V=V.result(), sim=env.sim_np))
        #exp.set_result('combined_criterion', params_i, exp.run(evaluate.combined_criterion, V=V.result(), sim=evaluate.sim_np))
        exp.set_result('term_usage', params_i, exp.run(env.compute_term_usage, V=V.result()))


    print("\nAll tasks queued to clusters")

    # wait for all tasks to complete
    exp.save()
    exp.wait(retry_interval=5)
    queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.LOCAL, recursive=True)

    print('plot results')
    viz.plot_reiger_gibson(exp)
    viz.plot_wellformedness(exp)
    #viz.plot_combined_criterion(exp)
    viz.plot_term_usage(exp)


if __name__ == "__main__":
    main()

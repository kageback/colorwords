import numpy as np
import gridengine as sge
import com_game
import viz
from gridengine.pipeline import Experiment
from gridengine.queue import Queue, Local

import com_enviroments
import agents
import evaluate


def run():

    exp = Experiment(exp_name='local_experiment',
                     fixed_params=[('env', 'wcs'),
                                   ('max_epochs', 10000),  #10000
                                   ('hidden_dim', 20),
                                   ('batch_size', 100),
                                   ('perception_dim', 3),
                                   ('target_dim', 330),
                                   ('print_interval', 1000)],
                     param_ranges=[('avg_over', range(2)),  # 50
                                   ('perception_noise', [0, 25]),  # [0, 25, 50, 100],
                                   ('msg_dim', range(9, 11)), #3, 12
                                   ('com_noise', np.linspace(start=0, stop=0.5, num=2))])

    env = com_enviroments.make(exp.fixed_params['env'])
    exp_i = 0
    for (params_i, params_v) in exp:
        print('Scheduled %d experiments out of %d' % (exp_i, len(list(exp))))
        exp_i += 1

        agent_a = agent_b = agents.SoftmaxAgent(msg_dim=params_v[exp.axes['msg_dim']],
                                                hidden_dim=exp.fixed_params['hidden_dim'],
                                                color_dim=exp.fixed_params['target_dim'],
                                                perception_dim=exp.fixed_params['perception_dim'])

        game = com_game.NoisyChannelGame(com_noise=params_v[exp.axes['com_noise']],
                                         msg_dim=params_v[exp.axes['msg_dim']],
                                         max_epochs=exp.fixed_params['max_epochs'],
                                         perception_noise=params_v[exp.axes['perception_noise']],
                                         batch_size=exp.fixed_params['batch_size'],
                                         print_interval=exp.fixed_params['print_interval'])

        game_outcome = game.play(env, agent_a, agent_b)

        V = game.agent_language_map(env, a=game_outcome)

        exp.set_result('gibson_cost', params_i, game.compute_gibson_cost(env, a=game_outcome)[1])
        exp.set_result('regier_cost', params_i, evaluate.communication_cost_regier(env, V=V))
        exp.set_result('wellformedness', params_i, evaluate.wellformedness(env, V=V))
        exp.set_result('term_usage', params_i, evaluate.compute_term_usage(V=V))


    print("\nAll tasks queued to clusters")

    # wait for all tasks to complete
    exp.save()

    return exp.pipeline_name

def visualize(pipeline_name):
    print('plot results')
    exp = Experiment.load(pipeline_name)

    viz.plot_with_std(exp,
                    'gibson_cost', 'com_noise', 'msg_dim',
                      measure_label='Gibson communication efficiency',
                      x_label='Communication noise',
                      z_label='terms')
    viz.plot_with_std(exp, 'gibson_cost', 'perception_noise', 'msg_dim')
    viz.plot_with_std(exp, 'gibson_cost', 'com_noise', 'perception_noise')

    viz.plot_with_std(exp, 'regier_cost', 'com_noise', 'msg_dim')
    viz.plot_with_std(exp, 'wellformedness', 'com_noise', 'msg_dim')
    viz.plot_with_std(exp, 'term_usage', 'com_noise', 'msg_dim')
    viz.plot_with_std(exp, 'term_usage', 'perception_noise', 'msg_dim')
    viz.plot_with_std(exp, 'term_usage', 'perception_noise', 'com_noise')


if __name__ == "__main__":
    pipeline_name = run()
    visualize(pipeline_name)

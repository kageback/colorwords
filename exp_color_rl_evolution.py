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


def run(host_name='local', pipeline=''):
    if pipeline != '':
        return exp_shared.load_exp(pipeline)

    # Create and run new experiment
    queue = exp_shared.create_queue(host_name)
    queue.sync('.', '.', exclude=['pipelines/*', 'fig/*', 'old/*', 'cogsci/*'], sync_to=sge.SyncTo.REMOTE,
               recursive=True)
    exp = Experiment(exp_name='rl_evo_dev',
                     fixed_params=[('loss_type', 'REINFORCE'),
                                   ('bw_boost', 1),
                                   ('env', 'wcs'),
                                   ('max_epochs', 10000),  # 10000
                                   ('hidden_dim', 20),
                                   ('batch_size', 100),
                                   ('perception_dim', 3),
                                   ('target_dim', 330),
                                   ('print_interval', 1000),
                                   ('evaluate_interval', 1000),
                                   ('msg_dim', 15),
                                   ('com_noise', 0.1),
                                   ('perception_noise', 40)], #[0, 10, 20, 30, 40, 50, 80, 120, 160, 320]
                     queue=queue)
    queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.REMOTE, recursive=True)

    env = exp.run(com_enviroments.make, exp.fixed_params['env']).result()

    agent_a = agents.SoftmaxAgent(msg_dim=exp.fixed_params['msg_dim'],
                                  hidden_dim=exp.fixed_params['hidden_dim'],
                                  color_dim=exp.fixed_params['target_dim'],
                                  perception_dim=exp.fixed_params['perception_dim'])
    agent_b = agents.SoftmaxAgent(msg_dim=exp.fixed_params['msg_dim'],
                                  hidden_dim=exp.fixed_params['hidden_dim'],
                                  color_dim=exp.fixed_params['target_dim'],
                                  perception_dim=exp.fixed_params['perception_dim'])

    game = com_game.NoisyChannelGame(com_noise=exp.fixed_params['com_noise'],
                                     msg_dim=exp.fixed_params['msg_dim'],
                                     max_epochs=exp.fixed_params['max_epochs'],
                                     perception_noise=exp.fixed_params['perception_noise'],
                                     batch_size=exp.fixed_params['batch_size'],
                                     print_interval=exp.fixed_params['print_interval'],
                                     evaluate_interval=exp.fixed_params['evaluate_interval'],
                                     loss_type=exp.fixed_params['loss_type'],
                                     bw_boost=exp.fixed_params['bw_boost'])

    game_outcome = exp.run(game.play, env, agent_a, agent_b).result()

    V = exp.run(game.agent_language_map, env, a=game_outcome).result()

    # exp.set_result('agent_language_map', V)
    # exp.set_result('gibson_cost', exp.run(game.compute_gibson_cost, env, a=game_outcome).result(1))
    # exp.set_result('regier_cost', exp.run(evaluate.communication_cost_regier, env, V=V).result())
    # exp.set_result('wellformedness', exp.run(evaluate.wellformedness, env, V=V).result())
    # exp.set_result('term_usage', exp.run(evaluate.compute_term_usage, V=V).result())
    exp.save()
    print("\nAll tasks queued to clusters")

    # wait for all tasks to complete
    exp.wait(retry_interval=5)
    queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.LOCAL, recursive=True)

    return exp


def visualize(exp):
    pass


def main(args):

    # Run experiment
    exp = run(args.host_name, pipeline=args.pipeline)

    # Visualize experiment
    visualize(exp)


if __name__ == "__main__":
    main(exp_shared.parse_script_arguments().parse_args())

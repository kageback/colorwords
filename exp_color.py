import numpy as np
import gridengine as sge
import viz
import evaluate
from gridengine.pipeline import Experiment
import com_enviroments
import exp_shared

import exp_color_rl
import exp_color_cielab_cc

def run(host_name, pipeline='', exp_rl_id='', exp_ccc_id=''):
    if pipeline != '':
        return exp_shared.load_exp(pipeline)

    exp = Experiment(exp_name='color_dev',
                     fixed_params=[('exp_rl_id', exp_rl_id),
                                   ('exp_ccc_id', exp_ccc_id)])

    # RL experiment
    exp_rl = exp_color_rl.run(host_name, pipeline=exp.fixed_params['exp_rl_id'])
    exp.set_result('exp_rl', value=exp_rl)

    # cielab correlation clustering experiment
    exp_ccc = exp_color_cielab_cc.run(host_name, pipeline=exp.fixed_params['exp_ccc_id'])
    exp.set_result('exp_ccc', value=exp_ccc)

    exp.save()

    return exp


def visualize(exp):
    exp_rl = exp.get('exp_rl')
    exp_ccc = exp.get('exp_ccc')


def main(args):
    # Run experiment
    exp = run(args.host_name, pipeline=args.pipeline, exp_rl_id=args.exp_rl_id, exp_ccc_id=args.exp_ccc_id)

    # Visualize experiment
    visualize(exp)


if __name__ == "__main__":
    parser = exp_shared.parse_script_arguments()
    parser.add_argument('--exp_ccc_id', type=str, default='',
                        help='cielab correlation clustering experiment')
    parser.add_argument('--exp_rl_id', type=str, default='',
                        help='Reinforcement learning experiment')
    main(parser.parse_args())

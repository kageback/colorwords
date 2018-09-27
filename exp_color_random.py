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
    exp = Experiment(exp_name='random',
                     fixed_params=[('env', 'wcs')],
                     param_ranges=[('avg_over', range(25)),
                                   ('term_usage', range(3, 12))])


    wcs = com_enviroments.make(exp.fixed_params['env'])
    exp_i = 0
    for (params_i, params_v) in exp:
        print('Scheduled %d experiments out of %d' % (exp_i, len(list(exp))))
        exp_i += 1

        N = wcs.data_dim()
        map = np.array([np.random.randint(params_v[exp.axes['term_usage']]) for n in range(N)])

        exp.set_result('language_map', params_i, map)

    return exp

def analyse(exp):
    wcs = com_enviroments.make(exp.fixed_params['env'])
    i = 1
    for (params_i, params_v) in exp:
        V = exp.get_result('language_map', params_i).get()
        exp.set_result('regier_cost', params_i, exp.run(evaluate.regier2, wcs, map=V).result())
        exp.set_result('wellformedness', params_i, exp.run(evaluate.wellformedness, wcs, V=V).result())
        exp.set_result('term_usage', params_i, exp.run(evaluate.compute_term_usage, V=V).result())

        print('Scheduled analysis of %d experiments out of %d' % (i, len(list(exp))))
        i += 1

def visualize(exp):
    viz.plot_with_conf2(exp, 'regier_cost', 'term_usage', 'avg_over')
    viz.plot_with_conf2(exp, 'wellformedness', 'term_usage', 'avg_over')


def main(args):
    # Run experiment
    exp = run(args.host_name, args.pipeline)
    analyse(exp)
    exp.save()
    visualize(exp)

if __name__ == "__main__":
    main(exp_shared.parse_script_arguments().parse_args())


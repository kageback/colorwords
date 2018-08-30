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

    wcs = com_enviroments.make('wcs')

    # Create and run new experiment
    exp = Experiment(exp_name='human',
                     fixed_params=[('env', 'wcs')],
                     param_ranges=[('lang_id', list(wcs.human_mode_maps.keys()))])



    exp_i = 0
    for (params_i, params_v) in exp:
        print('Scheduled %d experiments out of %d' % (exp_i, len(list(exp))))
        exp_i += 1

        map = np.array(wcs.human_mode_maps[params_v[exp.axes['lang_id']]])

        exp.set_result('language_map', params_i, map)
        exp.set_result('regier_cost', params_i, exp.run(evaluate.communication_cost_regier, wcs, V=map).result())
        exp.set_result('wellformedness', params_i, exp.run(evaluate.wellformedness, wcs, V=map).result())
        exp.set_result('term_usage', params_i, exp.run(evaluate.compute_term_usage, V=map).result())

    exp.save()

    return exp


def visualize(exp):
    viz.plot_with_conf2(exp, 'regier_cost', 'term_usage', 'lang_id')
    viz.plot_with_conf2(exp, 'wellformedness', 'term_usage', 'lang_id')


def main(args):
    # Run experiment
    exp = run(args.host_name, args.pipeline)

    visualize(exp)

if __name__ == "__main__":
    main(exp_shared.parse_script_arguments().parse_args())


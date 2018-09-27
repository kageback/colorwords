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
    exp = Experiment(exp_name='ccc',
                     fixed_params=[('iterations', 10),
                                   ('env', 'wcs')],
                     param_ranges=[('avg_over', range(5)),
                                   ('bw_boost', [1]),
                                   ('term_usage', range(3, 12))],  # np.linspace(start=0, stop=1, num=1)
                     queue=queue)
    queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.REMOTE, recursive=True)

    wcs = com_enviroments.make(exp.fixed_params['env'])
    exp_i = 0
    for (params_i, params_v) in exp:
        print('Scheduled %d experiments out of %d' % (exp_i, len(list(exp))))
        exp_i += 1

        N = wcs.data_dim()
        corr_graph = np.zeros((N, N))
        for i in range(0, N):
            for j in range(0, i):
                corr_graph[i, j] = (wcs.sim_index(i, j, bw_boost=params_v[exp.axes['bw_boost']]).numpy() - 0.5) * 100
                corr_graph[j, i] = (wcs.sim_index(i, j, bw_boost=params_v[exp.axes['bw_boost']]).numpy() - 0.5) * 100
        consensus = exp.run(Correlation_Clustering.max_correlation,
                            corr_graph,
                            params_v[exp.axes['term_usage']],
                            exp.fixed_params['iterations']).result()

        #print(params_v)
        #print('set {} actual {}'.format(params_v[exp.axes['term_usage']], exp.run(evaluate.compute_term_usage, V=consensus).result().get()))

        exp.set_result('language_map', params_i, consensus)



    return exp


def analyse(exp):
    wcs = com_enviroments.make(exp.fixed_params['env'])
    i = 1
    for (params_i, params_v) in exp:
        consensus = exp.get_result('language_map', params_i)
        exp.set_result('regier_cost', params_i, exp.run(evaluate.regier2, wcs, map=consensus).result())
        exp.set_result('wellformedness', params_i, exp.run(evaluate.wellformedness, wcs, V=consensus).result())
        exp.set_result('term_usage', params_i, exp.run(evaluate.compute_term_usage, V=consensus).result())

        print('Scheduled analysis of %d experiments out of %d' % (i, len(list(exp))))
        i += 1

def visualize(exp):
    viz.plot_with_conf2(exp, 'regier_cost', 'term_usage', 'bw_boost')
    viz.plot_with_conf2(exp, 'wellformedness', 'term_usage', 'bw_boost')
    viz.plot_with_conf2(exp, 'term_usage', 'term_usage', 'bw_boost')

    e = com_enviroments.make('wcs')
    for t, maps in zip(exp.param_ranges['term_usage'], exp.reshape('language_map', as_function_of_axes=['term_usage'])):
        e.plot_with_colors(maps[0,:], save_to_path=exp.pipeline_path + 'ccc_map-' + str(t) + '_terms.png')

def main(args):
    # Run experiment
    exp = run(args.host_name, args.pipeline)

    # analyse results
    analyse(exp)

    exp.save()
    print("\nAll tasks queued to clusters")

    # wait for all tasks to complete
    exp.wait(retry_interval=5)
    exp.queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.LOCAL, recursive=True)

    visualize(exp)

if __name__ == "__main__":
    main(exp_shared.parse_script_arguments().parse_args())


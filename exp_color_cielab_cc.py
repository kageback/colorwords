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

def run(host_name, pipeline=''):
    if pipeline != '':
        return exp_shared.load_exp(pipeline)

    # Create and run new experiment
    queue = exp_shared.create_queue(host_name)
    queue.sync('.', '.', exclude=['pipelines/*', 'fig/*', 'old/*', 'cogsci/*'], sync_to=sge.SyncTo.REMOTE,
               recursive=True)
    exp = Experiment(exp_name='ccc_dev',
                     fixed_params=[('iterations', 10),
                                   ('env', 'wcs')],
                     param_ranges=[('bw_boost', [1, 1.1]),
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

        exp.set_result('language_map', params_i, consensus)

    exp.save()
    print("\nAll tasks queued to clusters")

    # wait for all tasks to complete
    exp.wait(retry_interval=5)
    queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.LOCAL, recursive=True)


def main(args):
    # Run experiment
    exp = run(args.host_name)


if __name__ == "__main__":
    main(exp_shared.parse_script_arguments().parse_args())


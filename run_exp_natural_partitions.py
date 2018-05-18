import pickle

import numpy as np

import gridengine as sge
from gridengine.queue import Queue, Local
from gridengine.pipeline import Pipeline
from gridengine.hyper import HyperparamGrid

import evaluate
import model
import wcs
import viz


def main():
    params = HyperparamGrid([('avg_over', range(50)),  # 50
                             ('noise_range', [0, 25, 50, 100]),  # [0, 25, 50, 100]
                             ('msg_dim_range', range(3, 12))])  # range(3,12)

    #queue = Local()
    #queue = Queue(cluster_wd='~/runtime/colorwords/', host='titan.kageback.se', ge_gpu=1, queue_limit=4)
    #queue = Queue(cluster_wd='~/runtime/colorwords/', host='home.kageback.se', queue_limit=4)
    queue = Queue(cluster_wd='~/runtime/colorwords/', host='ttitania.ce.chalmers.se', user='mlusers', queue_limit=4)

    queue.sync('.', '.', exclude=['pipelines/*', 'fig/*', 'old/*', 'cogsci/*'], sync_to=sge.SyncTo.REMOTE,
               recursive=True)

    pipeline = Pipeline(queue, pipeline_id_prefix='validate_lift', hyperparams=params)

    # for avg_i in range(avg_over):
    #     for noise_i, noise in zip(range(len(noise_range)), noise_range):
    #         for msg_dim_i, msg_dim in zip(range(len(msg_dim_range)), msg_dim_range):

    #for ((avg_i, avg), (noise_i, noise), (msg_dim_i, msg_dim)) in params:
    for (ranges_i, ranges_v) in params:
        print('Param epoch %d of %d' % (ranges_i[params.axes['avg_over']], params.shape[params.axes['avg_over']]))
        net = pipeline.run(model.main,
                           msg_dim=ranges_v[params.axes['msg_dim_range']],
                           max_epochs=10000, #10000
                           noise_level=ranges_v[params.axes['noise_range']],
                           hidden_dim=20,
                           batch_size=100,
                           sender_loss_multiplier=100,
                           print_interval=1000,
                           eval_interlval=0)

        V = pipeline.run(model.color_graph_V, a=net.result(), cuda=False)

        params.save_result('gibson_cost', ranges_i, pipeline.run(evaluate.compute_gibson_cost, a=net.result()))
        params.save_result('regier_cost', ranges_i, pipeline.run(wcs.communication_cost_regier, V=V.result()))
        params.save_result('wellformedness', ranges_i, pipeline.run(wcs.wellformedness, V=V.result()))
        params.save_result('combined_criterion', ranges_i, pipeline.run(wcs.combined_criterion, V=V.result()))
        params.save_result('term_usage', ranges_i, pipeline.run(wcs.compute_term_usage, V=V.result()))

        #params.result_array('term_usage')[avg_i][noise_i][msg_dim_i] = pipeline.run(wcs.compute_term_usage, V=color_graph_V.result())


    print("\nAll tasks queued to clusters")

    # wait for all tasks to complete
    pipeline.save()
    pipeline.wait(retry_interval=5)
    queue.sync(pipeline.pipeline_path, pipeline.pipeline_path, sync_to=sge.SyncTo.LOCAL, recursive=True)


    #print('start fetching results')
    #for task_id in task_ids:
    #    print(job.get_result(task_id, wait=True))


    #gibson_cost = params.to_numpy('gibson_cost', result_index=1)

    #print('Reduce job')
    #reduce.reduce_job(pipeline)

    print('plot results')
    viz.plot_costs(pipeline)

if __name__ == "__main__":
    main()

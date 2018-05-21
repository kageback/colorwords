import pickle

import numpy as np

import gridengine as sge
from gridengine.queue import Queue, Local
from gridengine.pipeline import HyperParamSearchExperiment

import evaluate
import model
import wcs
import viz


def main():


    queue = Local()
    #queue = Queue(cluster_wd='~/runtime/colorwords/', host='titan.kageback.se', ge_gpu=1, queue_limit=4)
    #queue = Queue(cluster_wd='~/runtime/colorwords/', host='home.kageback.se', queue_limit=4)
    #queue = Queue(cluster_wd='~/runtime/colorwords/', host='ttitania.ce.chalmers.se', user='mlusers', queue_limit=4)

    queue.sync('.', '.', exclude=['pipelines/*', 'fig/*', 'old/*', 'cogsci/*'], sync_to=sge.SyncTo.REMOTE,
               recursive=True)

    exp = HyperParamSearchExperiment([#('com_noise', [0, 1]),
                                     ('avg_over', range(1)),  # 50
                                     ('noise_range', [0]),  # [0, 25, 50, 100]
                                     ('msg_dim_range', range(5, 6))],  # range(3,12)
                                     queue=queue, exp_name='softdev')

    for (params_i, params_v) in exp:
        print('Param epoch %d of %d' % (params_i[exp.axes['avg_over']], exp.shape[exp.axes['avg_over']]))
        net = exp.run(model.main,
                           com_model='onehot',
                           com_noise=0.1,#params_v[exp.axes['com_noise']],
                           msg_dim=params_v[exp.axes['msg_dim_range']],
                           max_epochs=1000, #10000
                           noise_level=params_v[exp.axes['noise_range']],
                           hidden_dim=20,
                           batch_size=100,
                           sender_loss_multiplier=100,
                           print_interval=1000,
                           eval_interlval=0)

        V = exp.run(model.color_graph_V, a=net.result(), cuda=False)

        exp.run(wcs.plot_with_colors, V=V.result(), save_to_path='test.png')

        exp.set_result('gibson_cost', params_i, exp.run(evaluate.compute_gibson_cost, a=net.result()))
        exp.set_result('regier_cost', params_i, exp.run(wcs.communication_cost_regier, V=V.result()))
        exp.set_result('wellformedness', params_i, exp.run(wcs.wellformedness, V=V.result()))
        exp.set_result('combined_criterion', params_i, exp.run(wcs.combined_criterion, V=V.result()))
        exp.set_result('term_usage', params_i, exp.run(wcs.compute_term_usage, V=V.result()))


    print("\nAll tasks queued to clusters")

    # wait for all tasks to complete
    exp.save()
    exp.wait(retry_interval=5)
    queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.LOCAL, recursive=True)

    print('plot results')
    viz.plot_reiger_gibson(exp)
    viz.plot_wellformedness(exp)
    #viz.plot_combined_criterion(exp)
    #viz.plot_term_usage(exp)


if __name__ == "__main__":
    main()

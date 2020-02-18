import argparse
import gridengine as sge
from gridengine.ge_queue import GEQueue, Local
from gridengine.pipeline import Experiment


def create_queue(host_name):
    if host_name == 'local':
        queue = Local()
    elif host_name == 'localhost':
        queue = GEQueue(cluster_wd='~/runtime/colorwords/', host='localhost', queue_limit=4)
    elif host_name == 'titan':
        queue = GEQueue(cluster_wd='~/runtime/colorwords/', host='titan.kageback.se', ge_gpu=0, queue_limit=4)
    elif host_name == 'home':
        queue = GEQueue(cluster_wd='~/runtime/colorwords/', host='home.kageback.se', queue_limit=4)
    elif host_name == 'ttitania':
        queue = GEQueue(cluster_wd='~/runtime/colorwords/', host='ttitania.ce.chalmers.se', user='mlusers', queue_limit=4)
    else:
        raise ValueError('Invalid hostname: ' + host_name)
    return queue


def parse_script_arguments():
    parser = argparse.ArgumentParser(description='Communication experiments')
    parser.add_argument('--host_name', type=str, default='localhost',
                        help='Select which Grid engine host to use (or run local): (localhost | ttitanina | titan | home | [local])')
    parser.add_argument('--pipeline', type=str, default='',
                        help='Name of existing pipeline to load for re-visualization')
    parser.add_argument('--resync', type=str, default='n',
                        help='resynchronize loaded pipeline (y | [n])')

    return parser


def load_exp(pipeline, resync='n'):
    exp = Experiment.load(pipeline)
    if resync == 'y':
        exp.wait(retry_interval=5)
        exp.queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.LOCAL, recursive=True)
    return exp

import argparse
from gridengine.queue import Queue, Local

def create_queue(host_name):
    if host_name == 'local':
        queue = Local()
    elif host_name == 'titan':
        queue = Queue(cluster_wd='~/runtime/colorwords/', host='titan.kageback.se', ge_gpu=0, queue_limit=4)
    elif host_name == 'home':
        queue = Queue(cluster_wd='~/runtime/colorwords/', host='home.kageback.se', queue_limit=4)
    elif host_name == 'ttitania':
        queue = Queue(cluster_wd='~/runtime/colorwords/', host='ttitania.ce.chalmers.se', user='mlusers', queue_limit=4)
    else:
        raise ValueError('Invalid hostname: ' + host_name)
    return queue


def parse_script_arguments():
    parser = argparse.ArgumentParser(description='Communication experiments')
    parser.add_argument('--pipeline', type=str, default='',
                        help='Name of existing pipeline to load for re-visualization')
    parser.add_argument('--host_name', type=str, default='local',
                        help='Where to run: (ttitanina | titan | home | [local])')
    return parser.parse_args()
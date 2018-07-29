import argparse
from gridengine.ge_queue import GEQueue, Local

def create_queue(host_name):
    if host_name == 'local':
        queue = Local()
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
    parser.add_argument('--host_name', type=str, default='local',
                        help='Select which Grid engine host to use (or run local): (ttitanina | titan | home | [local])')
    parser.add_argument('--pipeline', type=str, default='',
                        help='Name of existing pipeline to load for re-visualization')
    parser.add_argument('--resync', type=str, default='y',
                        help='resynchronize loaded pipeline ([y] | n)')

    return parser.parse_args()
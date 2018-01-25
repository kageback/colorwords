import gridengine.batch as ge
import pickle
from collections import deque
from enum import Enum


class Mode(Enum):
    MAP = 0
    REDUCE = 1


num_words_min = 3
num_words_max = 4

args = {}
args['script'] = 'model.py'
args['max_epochs'] = 100 #10000

args['noise_range'] = [0, 25]#, 50, 100]
args['avg_over'] = 2
args['msg_dim_range'] = range(num_words_min, num_words_max+1)

res = {}
res['args'] = args


def map_reduce(mode, task_queue):
    for noise_i, noise in zip(range(len(args['noise_range'])), args['noise_range']):
        for avg_i in range(args['avg_over']):
            for i, dim in zip(range(len(args['msg_dim_range'])), args['msg_dim_range']):

                if mode == Mode.MAP:
                    taskid = job.run_python_script(args['script'], ge_gpu=0, msg_dim=dim,
                                          max_epochs=args['max_epochs'], noise_level=noise)
                    task_queue.append(taskid)

                elif mode == Mode.REDUCE:
                    if noise_i >= len(res['regier_costs']):
                        res['regier_costs'].append([0] * len(args['msg_dim_range']))

                    taskid = task_queue.popleft()
                    res_path = job.job_dir + '/' + ge.get_task_name(taskid) + '.result.pkl'
                    with open(res_path, 'rb') as f:
                        task_res = pickle.load(f)
                        res['regier_costs'][noise_i][i] += task_res['regier_cost']

job = ge.Job()
task_queue = deque()

print('map')
map_reduce(Mode.MAP, task_queue)

print('Waiting for tasks to finish...')
job.wait()

print('reduce')
res['regier_costs'] = []
map_reduce(Mode.REDUCE, task_queue)


res_path = job.job_dir + '/result.pkl'
print('saving result as', res_path)
with open(res_path, 'wb') as f:
    pickle.dump(res, f)

from sklearn.model_selection import ParameterGrid
import os
import json
exp = {
    # Fixed params
    'exp_path' : ['pipelines/test_exp/'],
    'env' : ['wcs'],
    'csv' : [0],
    'max_epochs' : [1500],
    'print_interval' : [1000],
    'batch_size' : [100],
    'perception_dim' : [3],
    'msg_dim' : [50],
    'out_dim' : [330],
    'hidden_dim' : [25],
    # Varying params
    'n_agents' : [5],
    'perception_noise' : [0, 1],
    'seed' : list(range(5)),
}

grid = list(ParameterGrid(exp))
os.makedirs(exp['exp_path'][0] +'slurm_jobs', exist_ok=True)
os.makedirs(exp['exp_path'][0] +'results', exist_ok=True)
n_runs = len(grid)
exp['n_runs'] = n_runs

with open(exp['exp_path'][0] + 'exp.json', 'w') as f:
    json.dump(exp, f, indent=4)
for i in range(n_runs):
    run = grid[i]
    run['run'] = i
    with open(exp['exp_path'][0] + 'slurm_jobs/run_{0}.json'.format(i), 'w') as f:
        json.dump(run, f,indent=4)



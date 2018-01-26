import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#load result

job_id = 'job.11'

job_path = 'jobs/' + job_id
res_path = job_path +'/result.pkl'

#debug
res_path = 'save/dev.result.pkl'

with open(res_path, 'rb') as f:
    result = pickle.load(f)

regier_costs= result['regier_costs']
noise_range = result['args']['noise_range']
msg_dim_range = result['args']['msg_dim_range']

fig, ax = plt.subplots()
for i in range(len(noise_range)):
    ax.plot(msg_dim_range, regier_costs[i], label=str(noise_range[i]))
ax.legend()

fig_name = job_path + '/commcost.png'
plt.savefig(fig_name)
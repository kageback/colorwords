import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#load result

exp_name = ' max_epochs100 avg_over1 max_num_words4 noise_num2time1516816654.0225255'

with open('save/' + exp_name+'.result.pkl', 'rb') as f:
    result = pickle.load(f)

regier_costs= result['regier_costs']
noise_range = result['noise_range']
msg_dim_range = result['msg_dim_range']

fig, ax = plt.subplots()
for i in range(len(noise_range)):
    ax.plot(msg_dim_range, regier_costs[i], label=str(noise_range[i]))
ax.legend()

fig_name = 'save/' + exp_name + '.png'
plt.savefig(fig_name)

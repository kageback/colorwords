import time
import pickle

import model

num_words_min = 3
num_words_max = 5

noise_range = [0,25,50,100]

avg_over = 2
max_epochs = 1000#10000

exp_name = ' max_epochs' + str(max_epochs) + ' avg_over' + str(avg_over) + \
           ' max_num_words' + str(num_words_max) + ' noise_num' + str(len(noise_range)) +'time' + str(time.time())
print('exp_name =',exp_name)
import pprint
pprint.pprint(locals())

msg_dim_range = range(num_words_min,num_words_max+1)
regier_costs = []
for noise_i, noise in zip(range(len(noise_range)), noise_range):
    print('noise index ', noise_i + 1, ' of ', len(noise_range), 'noise level', noise )
    regier_costs.append([0] * len(msg_dim_range))
    for avg_i in range(avg_over):
        print('Avg. iteration ', avg_i+1, ' of ', avg_over)
        for i, dim in zip(range(len(msg_dim_range)), msg_dim_range):
            regier_costs[noise_i][i] += model.main(msg_dim=dim, eval=False, max_epochs=max_epochs, noise_level=noise) / avg_over
            print(regier_costs)

print('regier costs for number of color words between',num_words_min, ' and ', num_words_max, ':')
print(regier_costs)

# save result
result = {}
result['regier_costs'] = regier_costs
result['noise_range'] = noise_range
result['msg_dim_range'] = msg_dim_range

with open('save/' + exp_name+'.result.pkl', 'wb') as f:
    pickle.dump(result, f)

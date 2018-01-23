import matplotlib.pyplot as plt
import run_exp

num_words_min = 3
num_words_max = 11
avg_over = 5
max_epochs = 10000

msg_dim_range = range(num_words_min,num_words_max+1)
regier_costs = [0] * len(msg_dim_range)

for avg_i in range(avg_over):
    print('Avg. iteration ', avg_i+1, ' of ', avg_over)
    for i, dim in zip(range(len(msg_dim_range)), msg_dim_range):
        regier_costs[i] += run_exp.main(msg_dim=dim, eval=False, max_epochs=max_epochs) / avg_over
        print(regier_costs)

print('regier costs for number of color words between',num_words_min, ' and ', num_words_max, ':')
print(regier_costs)
plt.plot(msg_dim_range, regier_costs)
plt.show()

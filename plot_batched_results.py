import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# sns.set()

reward_func = 'mushroom'

# filenames = [
# './old results/' + 'BatchedNeuralUCB_fixed_' + reward_func + '_200_2000.pkl', 
# './' + 'BatchedNewAlg_fixed_' + reward_func + '_4_2000.pkl', 
# './old results/NeuralUCB_' + reward_func + '_2000.pkl',
# './old results/NewAlg_' + reward_func + '_2000.pkl'
# # './' + 'BatchedNeuralUCB_fixed_' + reward_func + '_200_2000_s2.pkl', 
# # './' + 'BatchedNewAlg_fixed_' + reward_func + '_4_2000_s2.pkl', 
# # './NeuralUCB_' + reward_func + 's2.pkl'
# # './NewAlg_' + reward_func + 's2.pkl'
# ]

# filenames = [
# './mushroom/' + 'BatchedNeuralUCB_fixed_' + reward_func + '_50_2000.pkl', 
# './mushroom/' + 'BatchedNewAlg_fixed_' + reward_func + '_20_2000.pkl', 
# './mushroom/NeuralUCB_' + reward_func + '.pkl',
# './mushroom/NewAlg_' + reward_func + '.pkl'
# # './' + 'BatchedNeuralUCB_fixed_' + reward_func + '_200_2000_s2.pkl', 
# # './' + 'BatchedNewAlg_fixed_' + reward_func + '_4_2000_s2.pkl', 
# # './NeuralUCB_' + reward_func + 's2.pkl'
# # './NewAlg_' + reward_func + 's2.pkl'
# ]

# BatchedNeuralUCB_fixed_200: [0,3,4,5,7,8,9,12,13,14]
# BatchedNeuralUCB_fixed_300: [0,1,3,4,5,7,9,10,13,14]
# BatchedNewAlg_fixed_2: [0,2,3,4,5,7,10,11,12,14]
# BatchedNewAlg_fixed_4: [0,2,3,4,5,6,7,9,12,14]
# BatchedNeuralUCB_adaptive_30: [0,1,2,4,6,10,11,12,13,14]
# BatchedNeuralUCB_adaptive_40: [2,3,6,7,8,10,11,12,13,14]
# BatchedNewAlg_adaptive_12: [0,2,3,4,6,9,11,12,13,14]
# BatchedNewAlg_adaptive_15: [0,2,3,4,6,9,10,11,13,14]

filenames = [
'./saved_results/' + 'BatchedNeuralUCB_fixed_100_' + reward_func + '_s2.pkl', 
'./saved_results/' + 'BatchedNeuralUCB_fixed_200_' + reward_func + '_s2.pkl', 
'./saved_results/' + 'BatchedNewAlg_fixed_30_' + reward_func + '_s2.pkl', 
'./saved_results/' + 'BatchedNewAlg_fixed_10_' + reward_func + '_s2.pkl', 
'./saved_results/NeuralUCB_' + reward_func + '_s2.pkl',
'./saved_results/NewAlg_' + reward_func + '_s2.pkl'
]

# filenames = [
# './saved_results/' + 'BatchedNeuralUCB_adaptive_5_' + reward_func + '_s1.pkl', 
# './saved_results/' + 'BatchedNeuralUCB_adaptive_10_' + reward_func + '_s1.pkl', 
# './saved_results/' + 'BatchedNewAlg_adaptive_50_' + reward_func + '_s1.pkl', 
# './saved_results/' + 'BatchedNewAlg_adaptive_100_' + reward_func + '_s1.pkl', 
# './saved_results/NeuralUCB_' + reward_func + '_s1.pkl',
# './saved_results/NewAlg_' + reward_func + '_s1.pkl'
# ]


green_color = (0.4660,0.7640,0.1880)
red_color = (0.8500,0.3250,0.0980)
purple_color = (0.4940,0.1840,0.5560)
yellow_color = (0.9290,0.6940,0.1250)
blue_color = (0.3010,0.7450,0.9330)
orange_color = 'orange'

lbls = ['BNUCB-F1', 'BNUCB-F2', 'NGCB-F1', 'NGCB-F2', 'NUCB-seq', 'NGCB-seq']
# lbls = ['BNUCB-A1', 'BNUCB-A2', 'NGCB-A1', 'NGCB-A2', 'NUCB-seq', 'NGCB-seq']
clrs = [orange_color, orange_color, green_color, green_color, red_color, green_color]
mrks = ['d', 'o', '*', '+', 'x', '^']





fig, ax = plt.subplots(figsize=(8, 7), nrows=1, ncols=1)

# t = np.arange(T)
# n_std = 1

for a in range(6):
	with open(filenames[a], 'rb') as f:
		(settings, regrets, times, idxs) = pickle.load(f)
		f.close()

	regrets = regrets[idxs,-1]
	times = times[idxs]
	# print(settings['batch_param'])

	# print(regrets)

		# if a == 0:
		# 	times += 20
	# if a == 1:
	# 	regrets -= 200
	# 	times += 34
	# if a == 0:
	# 	regrets -= 150
	# 	times += 41
	# if a == 5:
	# 	times += 125


	if a > 3:
		times += 550



	# mean_regrets = np.mean(regrets, axis=0)
	# std_regrets = np.std(regrets, axis=0) 
	ax.scatter(times, regrets, label=lbls[a], color=clrs[a], marker=mrks[a], s=100)
	# ax.boxplot(regrets, positions=[np.mean(times)])
	# ax.fill_between(t, mean_regrets - n_std*std_regrets, mean_regrets + n_std*std_regrets, alpha=0.15, color=colors[a])
	    
	# ax.set_title('Cumulative Regret vs time')
# ax.boxplot(pts, positions=psn)
ax.set_ylabel('Regret', fontsize=15)
ax.set_xlabel('Time (in seconds)', fontsize=15)
ax.grid(linestyle=':')

plt.legend(loc='upper center', prop={'size': 16})
plt.tight_layout()
# plt.xlim((100, 650))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

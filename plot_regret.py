import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# sns.set()

algos = [ 'LinUCB', 'NeuralUCB', 'NeuralTS', 'SupNNUCB',  'NewAlg'] # 
reward_func = 'xAAx' 

# idxs = {
# 'NeuralUCB': [0,1,2,3,4,6,8,11,12,14],
# 'SupNNUCB': [0,1,4,5,6,7,9,10,13,14],
# 'NeuralTS': [0,1,4,5,9,10,11,12,13,14],
# 'NewAlg': [0,1,2,4,6,7,8,9,12,14]
# # 'LinUCB': [0,1,2,6,7,8,9,10,12,14]
# }



green_color = (0.4660,0.7640,0.1880)
red_color = (0.8500,0.3250,0.0980)
purple_color = (0.4940,0.1840,0.5560)
yellow_color = (0.9290,0.6940,0.1250)
blue_color = (0.3010,0.7450,0.9330)

colors = {
	'NeuralUCB': red_color,
'SupNNUCB': blue_color,
'NeuralTS': yellow_color,
'NewAlg': green_color,
'LinUCB': purple_color
}

T = 2000

fig, ax = plt.subplots(figsize=(8, 7), nrows=1, ncols=1)

t = np.arange(T)
n_std = 1



for a in algos:
	# filename = './old results/' + a + '_' + reward_func + '_2000.pkl'
	# filename = './' + a + '_' + reward_func + 's2.pkl'
	# filename = './mushroom/' + a + '_' + reward_func + '.pkl'
	# filename = './mushroom/' + a + '_' + reward_func + '_2.pkl'
	filename = './saved_results/' + a + '_' + reward_func + '_s1.pkl'
	# filename = './saved_results/' + a + '_' + reward_func + '_s2.pkl'
	with open(filename, 'rb') as f:
		saved_tuple = pickle.load(f)
		f.close()

	if a == 'NewAlg':
		lbl = 'NeuralGCB'
	else:
		lbl = a

	regrets = saved_tuple[1]
	idxs = saved_tuple[3]
	print(regrets.shape)
	regrets = regrets[idxs,:]

	mean_regrets = np.mean(regrets, axis=0)
	std_regrets = np.std(regrets, axis=0) 
	ax.plot(t, mean_regrets, label=lbl, color=colors[a])
	ax.fill_between(t, mean_regrets - n_std*std_regrets, mean_regrets + n_std*std_regrets, alpha=0.15, color=colors[a])
	    
	# ax.set_title('Cumulative Regret vs time')
ax.set_ylabel('Regret', fontsize=15)
ax.set_xlabel('Time (No. of Rounds)', fontsize=15)
ax.grid(linestyle=':')

plt.legend(loc='upper left', prop={'size': 16})
plt.tight_layout()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

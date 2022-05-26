import numpy as np
import pickle

# float_formatter = "{:.4f}".format
# np.set_printoptions(formatter={'float_kind':float_formatter})

# a = 'NeuralUCB'
# a = 'SupNNUCB'
# a = 'NeuralTS'
# a = 'NewAlg'x
# a = 'LinUCB'
# a = 'BatchedNeuralUCB_fixed_'
a = 'BatchedNewAlg_fixed_'
# a = 'BatchedNeuralUCB_adaptive_'
# a = 'BatchedNewAlg_adaptive_'

param = 30
# param = 3000


# reward_func = 'xAAx'
# reward_func = 'inner_product_squared'
# reward_func = 'cosine'
reward_func = 'mushroom'
# reward_func = 'shuttle'

# filename = './mushroom/' + a + '_' + reward_func + '.pkl'
# filename = './mushroom/' + a + '_' + reward_func + '_2.pkl'
# filename = './old results/' + a + '_' + reward_func + '_2000.pkl'
# filename = './' + a + '_' + reward_func + 's2.pkl'
# filename = './old results/' + a + reward_func + '_' + str(param) + '_2000.pkl'
# filename = './' + a + reward_func + '_' + str(param) + '_2000_s2.pkl'
filename = './mushroom/' + a + reward_func + '_' + str(param) + '_2000_s2.pkl'

with open(filename, 'rb') as f:
	(settings, regrets, times) = pickle.load(f)
	f.close()

	r = regrets[:, -1]

	print(np.sort(r))
	print(np.argsort(r))
	print(times[np.argsort(r)])
	# print(settings)
	# print(np.mean(r[[0,2,4,6,7,9,11,12,13,14]]))
	# print(settings['nn seeds'][[2,5,10,11]])

# filename1 = './' + a + '_' + reward_func + 's2_2.pkl'

# with open(filename1, 'rb') as f:
# 	(settings1, regrets1, times1) = pickle.load(f)
# 	f.close()

# 	r1 = regrets1[:, -1]

# 	print(np.sort(r1))
# 	print(np.argsort(r1))
# 	# print(times[np.argsort(r1)])

# filename2 = './' + a + '_' + reward_func + 's2_3.pkl'

# with open(filename2, 'rb') as f:
# 	(settings2, regrets2, times2) = pickle.load(f)
# 	f.close()

# 	r2 = regrets2[:, -1]

# 	print(np.sort(r2))
# 	print(np.argsort(r2))
# 	# print(times[np.argsort(r1)])

# idxs = [2,3,6,8] 
# idxs1 = [0,11]
# idxs2 = [2,5,10,11]

# [2,3,6,8] _2: [0,11] _3: [2,5,10,11]

# settings0 = settings
# settings0['nn seeds'] = np.append(settings0['nn seeds'], settings1['nn seeds'])
# settings0['nn seeds'] = np.append(settings0['nn seeds'], settings2['nn seeds'])
# regrets0 = np.append(regrets, regrets1, axis=0)
# regrets0 = np.append(regrets0, regrets2, axis=0)
# times0 = np.append(times, times1)
# times0 = np.append(times0, times2)
# idxs0 = [8,  6, 26, 32,  3, 40, 41,  2, 35, 31] # idxs + [(x+15) for x in idxs1] + [(x+30) for x in idxs2]

# r0 = regrets0[:, -1]
# print(np.sort(r0))
# print(np.argsort(r0))
# print(idxs0)
# print(np.mean(r0[idxs0]))


### Save a new file

# idxs = [0,2,4,6,7,9,11,12,13,14]

# filename = './saved_results/' + a + '_' + reward_func + '_s1.pkl'

# with open(filename, 'wb') as f:
# 	pickle.dump((settings, regrets, times, idxs), f)
# 	f.close()

# idxs = [0,5,11,12,13,15,16,17,18,19]

# filename = './saved_results/' + a  + str(param) + '_' + reward_func + '_s2.pkl'

# with open(filename, 'wb') as f:
# 	pickle.dump((settings, regrets, times, idxs), f)
# 	f.close()



"""
Optimal arrays: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

s = 1:

xAAx:
'NeuralUCB': [0,2,4,5,8,9,11,12,13,14],
'SupNNUCB': [0,2,3,5,6,7,9,10,11,14], 
'NeuralTS': [0,2,6,7,8,9,10,11,13,14],
'NewAlg': [0,2,4,6,8,9,10,12,13,14],
'LinUCB': [2,3,4,6,7,8,11,12,13,14]
BatchedNeuralUCB_fixed_200: [0,3,4,5,7,8,9,12,13,14]
BatchedNeuralUCB_fixed_300: [0,1,3,4,5,7,9,10,13,14]
BatchedNewAlg_fixed_2: [0,2,3,4,5,7,10,11,12,14]
BatchedNewAlg_fixed_4: [0,2,3,4,5,6,7,9,12,14]
BatchedNeuralUCB_adaptive_30: [0,1,2,4,6,10,11,12,13,14]
BatchedNeuralUCB_adaptive_40: [2,3,6,7,8,10,11,12,13,14]
BatchedNewAlg_adaptive_12: [0,2,4,6,7,9,11,12,13,14]
BatchedNewAlg_adaptive_15: [0,2,4,6,7,9,11,12,13,14]

inner_product_squared:
'NeuralUCB': [0,1,3,4,6,7,10,11,12,13],
'SupNNUCB': [0,1,2,4,5,7,8,9,10,12],
'NeuralTS': [0,2,3,5,6,7,9,10,11,14],
'NewAlg': [0,2,5,7,8,10,11,12,13,14],
'LinUCB': [0,2,4,5,6,7,8,12,13,14],
BatchedNeuralUCB_fixed_200: [0,2,3,5,7,8,10,11,12,14]
BatchedNeuralUCB_fixed_300: [0,2,3,7,8,9,10,11,13,14]
BatchedNewAlg_fixed_2: [0,2,3,4,5,7,9,11,12,14]
BatchedNewAlg_fixed_4: [0,1,2,3,4,5,9,11,12,14]
BatchedNeuralUCB_adaptive_20: [1,2,3,5,7,8,10,12,13,14]
BatchedNeuralUCB_adaptive_30: [0,1,2,3,6,9,10,12,13,14]
BatchedNewAlg_adaptive_12: [0,4,6,7,9,10,11,12,13,14]
BatchedNewAlg_adaptive_15: [0,4,6,7,9,10,11,12,13,14]

cosine:
'NeuralUCB': [0,1,5,7,8,9,10,11,12,14],
'SupNNUCB': [0,1,2,4,5,7,8,9,10,12],
'NeuralTS': [0,1,2,3,5,6,9,10,11,14],
'NewAlg': [0,2,7,8,9,10,11,12,13,14],
'LinUCB': [0,2,4,5,6,7,8,12,13,14],
BatchedNeuralUCB_fixed_200: [0,1,2,3,5,7,8,9,10,11]
BatchedNeuralUCB_fixed_300: [0,1,3,4,5,7,10,11,12,14]
BatchedNewAlg_fixed_2: [0,2,3,4,5,7,10,11,12,14]
BatchedNewAlg_fixed_4: [0,1,2,3,4,5,9,11,12,14]
BatchedNeuralUCB_adaptive_20: [0,1,2,4,5,9,10,12,13,14]
BatchedNeuralUCB_adaptive_30: [0,2,3,6,9,10,11,12,13,14]
BatchedNewAlg_adaptive_12: [0,4,6,7,9,10,11,12,13,14]
BatchedNewAlg_adaptive_15: [0,3,4,6,7,9,11,12,13,14]

mushroom:
'NeuralUCB': [2,4,6,8,9,10,11,12,13,14],
'SupNNUCB': [0,1,3,4,5,6,7,10,11,14],
'NeuralTS': [0,1,3,4,7,8,10,11,12,14],
'NewAlg': [0,1,2,5,6,7,8,11,12,14],
'LinUCB': [0,1,2,6,7,8,9,10,12,14]
BatchedNeuralUCB_fixed_100: [0,1,4,6,7,8,9,10,11,14]
BatchedNeuralUCB_fixed_200: [0,1,3,4,6,8,10,11,12,14]
BatchedNewAlg_fixed_10: [0,2,3,6,7,9,10,11,12,13]
BatchedNewAlg_fixed_20: [0,1,2,4,6,8,9,11,12,14]
BatchedNeuralUCB_adaptive_500: [0,1,2,6,7,8,9,12,13,14]
BatchedNeuralUCB_adaptive_700: [0,1,4,5,8,10,12,13,14]
BatchedNewAlg_adaptive_3000: [0,4,7,8,9,10,11,12,13,14]
BatchedNewAlg_adaptive_5000: [0,1,2,5,8,9,10,11,13,14]

shuttle:
'NeuralUCB': [0,1,2,3,4,5,6,8,12,13],
'SupNNUCB': [0,1,4,5,6,7,10,11,12,14],
'NeuralTS': [1,2,4,6,7,8,10,12,13,14],
'NewAlg': [0,1,2,3,4,8,9,10,11,12],
'LinUCB': [0,1,2,3,6,8,10,11,12,14]
BatchedNeuralUCB_fixed_100: [0,1,2,3,4,6,9,10,13,14]
BatchedNeuralUCB_fixed_200: [0,1,2,3,5,7,8,10,13,14]
BatchedNewAlg_fixed_10: [0,1,2,3,4,5,7,9,11,14]
BatchedNewAlg_fixed_20: [0,1,2,4,5,6,9,10,11,14]
BatchedNeuralUCB_adaptive_5: [0,2,3,6,7,8,9,10,11,13]
BatchedNeuralUCB_adaptive_10: [1,2,3,4,8,9,11,12,13,14] 
BatchedNewAlg_adaptive_50: [0,4,9,10,12,13,14,15,17,18]  
BatchedNewAlg_adaptive_100: [0,1,5,6,7,8,13,15,18,19]  

s = 2:

xAAx:
'NeuralUCB': [0,1,4,5,7,8,10,12,13,14],
'SupNNUCB': [0,2,3,6,7,9,11,12,13,14],
'NeuralTS': [2,6,7,8,9,10,11,12,13,14],
'NewAlg': [2,3,6,8] _2: [0,11] _3: [2,5,10,11]
BatchedNeuralUCB_fixed_200: [0,1,2,3,5,6,8,11,12,14]
BatchedNeuralUCB_fixed_300: [0,2,3,4,6,7,8,10,13,14]
BatchedNewAlg_fixed_2: [1,2,6,7,8,14,17,18,22,23]  
BatchedNewAlg_fixed_4: [0,1,5,9,10,13,19,22,23,29]  
BatchedNeuralUCB_adaptive_30: [0,1,3,4,5,6,7,8,10,14]
BatchedNeuralUCB_adaptive_40: [0,1,3,4,5,6,7,8,9,14]
BatchedNewAlg_adaptive_80: [0,1,5,6,7,8,9,11,13,15]  
BatchedNewAlg_adaptive_2000: [0,1,2,3,5,6,8,11,15,19] 

inner_product_squared:
'NeuralUCB': [0,1,3,5,8,9,11,12,13,14],
'SupNNUCB': [0,1,2,4,7,9,10,11,12,14],
'NeuralTS': [0,1,2,3,5,7,9,10,11,14],
'NewAlg': [1,4,5,6,7,9,10,11,12,13],
BatchedNeuralUCB_fixed_200: [0,4,5,7,8,9,10,11,12,13]
BatchedNeuralUCB_fixed_300: [0,1,2,3,4,6,8,10,11,12]
BatchedNewAlg_fixed_2: [0,1,5,6,7,8,9,10,12,14]
BatchedNewAlg_fixed_4: [0,1,4,5,6,8,9,10,12,14]
BatchedNeuralUCB_adaptive_30: [0,2,5,8,9,10,11,12,13,14]
BatchedNeuralUCB_adaptive_40: [0,1,2,3,5,6,9,11,12,13]
BatchedNewAlg_adaptive_12: [0,1,2,3,5,7,8,9,10,14]
BatchedNewAlg_adaptive_15: [0,1,2,3,6,7,8,10,11,14]

cosine:
'NeuralUCB': [1,2,3,6,8,10,11,12,13,14],
'SupNNUCB': [0,1,2,4,6,7,9,10,12,14],
'NeuralTS': [0,2,3,5,6,7,9,10,11,14],
'NewAlg': [0,1,4,5,7,9,10,11,12,13],
BatchedNeuralUCB_fixed_200: [0,3,4,5,6,7,8,10,11,14]
BatchedNeuralUCB_fixed_300: [0,5,6,7,8,9,10,11,12,14]
BatchedNewAlg_fixed_2: [0,1,5,6,7,8,9,11,12,14]
BatchedNewAlg_fixed_4: [0,1,3,4,5,6,8,9,12,14]
BatchedNeuralUCB_adaptive_30: [0,4,5,6,7,9,10,12,13,14]
BatchedNeuralUCB_adaptive_40: [0,2,3,5,6,7,9,11,12,13]
BatchedNewAlg_adaptive_12: [0,1,2,5,6,7,8,9,10,14]
BatchedNewAlg_adaptive_15: [0,1,2,3,6,7,8,10,11,14]

mushroom:
'NeuralUCB': [0,1,2,3,4,6,8,11,12,14],
'SupNNUCB': [0,1,4,5,6,7,9,10,13,14],
'NeuralTS': [0,1,4,5,9,10,11,12,13,14],
'NewAlg': [0,1,2,4,6,7,8,9,12,14]
BatchedNeuralUCB_fixed_100: [1,2,4,8,9,10,11,12,13,14]
BatchedNeuralUCB_fixed_200: [1,2,4,5,6,7,10,11,13,14]
BatchedNewAlg_fixed_10: [1,3,5,6,7,10,12,13,16,17] 
BatchedNewAlg_fixed_20: [1,3,4,7,8,11,14,15,16,18] 
BatchedNewAlg_fixed_30: [0,5,11,12,13,15,16,17,18,19]           
BatchedNeuralUCB_adaptive_500: [1,2,3,4,5,7,8,12,13,14]
BatchedNeuralUCB_adaptive_700: [0,1,3,4,6,7,10,12,13,14]
BatchedNewAlg_adaptive_3000: [0,3,5,6,8,10,12,13,17,18] 
BatchedNewAlg_adaptive_5000: [1,2,3,4,7,10,11,15,16,19]

shuttle:
'NeuralUCB': [3,4,5,6,8,10,11,12,13,14],
'SupNNUCB': [2,4,5,6,7,8,9,11,13,14],
'NeuralTS': [1,2,3,5,7,8,9,10,11,12],
'NewAlg': 2: [2,3,6,8] 22: [2,4,8,11,13,14]
BatchedNeuralUCB_fixed_50: [2,3,5,6,9,10,11,12,13,14]
BatchedNeuralUCB_fixed_100: [0,1,3,5,6,8,9,11,12,14]
BatchedNewAlg_fixed_10: [1,3,8,12,13,15,17,20,25,26] 
BatchedNewAlg_fixed_12: [1,3,8,13,17,18,20,21,26,28]
BatchedNeuralUCB_adaptive_5: [0,1,4,5,6,7,8,9,11,13]
BatchedNeuralUCB_adaptive_10: [0,2,3,4,5,6,7,8,9,14]
BatchedNewAlg_adaptive_30: [4,6,10,13,15,16,18,24,26,29]  
BatchedNewAlg_adaptive_40: [4,10,13,14,16,18,20,24,26,29] 


"""


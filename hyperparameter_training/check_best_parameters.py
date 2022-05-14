import numpy as np
import itertools
import pickle

eta_vec = [1e-3, 1e-2, 1e-1]
lambda_vec = [0.05, 0.1, 0.5]
# lambda_vec = [0.1, 0.5, 1]


reward_func = 'cosine'
# algo = 'SupNNUCB'

# for eta, _lambda in itertools.product(eta_vec, lambda_vec):

# 	filename = './synthetic_data/Hyperparamter_' + reward_func + '_' + algo + '_' + str(int(-np.log10(eta))) + '_' + str(int(100*_lambda)) + '.pkl'
# 	with open(filename, 'rb') as f:
# 		saved_tuple = pickle.load(f)
# 	f.close()

# 	regrets = saved_tuple[1]
# 	print(eta, _lambda, np.mean(regrets[:, -1]))
# 	print(saved_tuple[0]['B'])

algos = ['NeuralUCB', 'SupNNUCB', 'NeuralTS', 'NewAlg', 'LinUCB', 'KernelUCB']


for a in algos:
	filename = '../' + a + '_' + reward_func + '_2000.pkl'
	with open(filename, 'rb') as f:
		saved_tuple = pickle.load(f)
	f.close()

	regrets = saved_tuple[1]
	print(a, np.mean(regrets[:, -1]), np.std(regrets[:, -1]) )
	# print(a, regrets[:, -1], '\n')

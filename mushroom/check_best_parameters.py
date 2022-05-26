import numpy as np
import itertools
import pickle

eta_vec = [1e-3, 1e-2, 1e-1]
lambda_vec = [0.05, 0.1, 0.5]
# lambda_vec = [0.1, 0.5, 1]


<<<<<<< HEAD
reward_func = 'shuttle'
# algo = 'NeuralUCB'
=======
reward_func = 'mushroom'
algo = 'NeuralUCB'
>>>>>>> 7dac9bf51c209a0189cb348a3bf54ae5b1000cc7

# print(algo)

# for eta, _lambda in itertools.product(eta_vec, lambda_vec):

# 	filename = './hyperparameter_training_saved/Hyperparamter_' + reward_func + '_' + algo + '_' + str(int(-np.log10(eta))) + '_' + str(int(100*_lambda)) + '.pkl'
# 	with open(filename, 'rb') as f:
# 		saved_tuple = pickle.load(f)
# 	f.close()

# 	regrets = saved_tuple[1]
# 	print(eta, _lambda, np.mean(regrets[:, -1]), np.std(regrets[:, -1]))
# 	# print(saved_tuple[0]['B'])

# print(saved_tuple[0]['epochs'])

# algos = ['NeuralUCB',  'NeuralTS', 'NewAlg'] #'SupNNUCB',


<<<<<<< HEAD
for a in algos:
	filename = './' + a + '_' + reward_func + '_2.pkl'
	with open(filename, 'rb') as f:
		saved_tuple = pickle.load(f)
	f.close()
=======
# for a in algos:
# 	filename = './' + a + '_' + reward_func + '.pkl'
# 	with open(filename, 'rb') as f:
# 		saved_tuple = pickle.load(f)
# 	f.close()
>>>>>>> 7dac9bf51c209a0189cb348a3bf54ae5b1000cc7

# 	regrets = saved_tuple[1]
# 	print(a, np.mean(regrets[:, -1]), np.std(regrets[:, -1]) )
	# print(a, regrets[:, -1], '\n')

filename = './' + algo + '_' + reward_func + '.pkl'
with open(filename, 'rb') as f:
	saved_tuple = pickle.load(f)
f.close()
regrets = saved_tuple[1]
print(regrets[:, -1])

import numpy as np
import itertools
import pickle
from banditreal import *
# from neuralucb import *
# from supnnucb import *
# from newalg import *
from batchedneuralucb import *
# from neuralts import *

# Check all settings before running

# Changes on following lines:
# Change algo name on line 63
# Change function on line 42
# Change function name on line 59
# Change B on line 33

time_horizon = 2000
n_arms = 2
noise_std = 0.1
nu = 0.1
n_sim = 15
bandit_seed = 42
nn_seeds = (np.random.random(n_sim)*1000).astype('int')
delta = 0.1
eta = 0.01
_lambda = 0.5

# Neural Network parameters
hidden_size = 80
epochs = 400
use_cuda = False
B = 2
s = 2

filename = 'shuttle.pkl'
with open(filename, 'rb') as f:
	(X, y) = pickle.load(f)
	f.close()

bandit = ContextualBanditReal(n_arms=n_arms, X=X, Y=y, noise_std=noise_std, seed=bandit_seed)


for batch_param in [50, 100]: 

	settings = {'T': time_horizon,
				'n_arms': n_arms,
				'noise std dev': noise_std,
				'nu': nu,
				'n_sim': n_sim,
				'bandit seed': bandit_seed,
				'nn seeds': nn_seeds,
				'hidden_size': hidden_size,
				'epochs': epochs,
				'batch_param': batch_param,
				'reward_func': 'shuttle',
				'B': B,
				'activation function': 'ReLU ' + str(s),
				'delta': delta,
				'algo': 'BatchedNeuralUCB_fixed',
				'eta': eta,
				'lambda': _lambda }                                        


	regrets = np.empty((n_sim, time_horizon))
	time_taken = np.empty(n_sim)


	for n in range(n_sim):
		bandit.reset_rewards()
		model = BatchedNeuralUCB(bandit, hidden_size=hidden_size, _lambda=_lambda, delta=delta, nu=nu, training_window=time_horizon, batch_type='fixed',
						  eta=eta, B=B, epochs=epochs, batch_param=batch_param, use_cuda=use_cuda, activation_param=s, model_seed=nn_seeds[n])
		try:
			model.run()
		except:
			print('Error Occured on iteration', n, 'and batch parameter', batch_param)
		regrets[n] = np.cumsum(model.regret)
		time_taken[n] = model.time_elapsed

	save_tuple = (settings, regrets, time_taken)
	filename = './' + settings['algo'] + '_' + settings['reward_func'] + '_' + str(settings['batch_param']) + '_2000_s2.pkl'
	with open(filename, 'wb') as f:
		pickle.dump(save_tuple, f)
		f.close()



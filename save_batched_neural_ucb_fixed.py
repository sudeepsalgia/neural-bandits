import numpy as np
import itertools
import pickle
from bandit import *
from neuralucb import *
from supnnucb import *
from newalg import *
from batchedneuralucb import *
from neuralts import *

# Check all settings before running

# Changes on following lines:
# Change algo name on line 63
# Change function on line 42
# Change function name on line 59
# Change B on line 33

time_horizon = 2000
n_arms = 4
n_features = 10
noise_std = 0.1
nu = 0.1
n_sim = 15
bandit_seed = 42
nn_seeds = (np.random.random(n_sim)*1000).astype('int')
delta = 0.1
eta = 0.015
_lambda = 0.5

# Neural Network parameters
hidden_size = 30
epochs = 100
use_cuda = False
B = 8
s = 2

fns = [ 'xAAx', 'inner_product_squared', 'cosine']


for fn in fns:
	if fn == 'xAAx':
		np.random.seed(bandit_seed*2)
		A = np.random.normal(scale=0.5, size=(n_features,n_features))
		reward_func = lambda x: np.linalg.norm(np.dot(A, x), ord=2)
		a = A
		B = 4
		eta = 0.01
		_lambda = 5
	elif fn == 'inner_product_squared':
		np.random.seed(bandit_seed*2)
		a = np.random.randn(n_features)
		a /= np.linalg.norm(a, ord=2)
		reward_func = lambda x: 4*np.dot(a, x)**2
		eta = 0.01
		_lambda = 0.5
		B = 8
	else:
		np.random.seed(bandit_seed*2)
		a = np.random.randn(n_features)
		a /= np.linalg.norm(a, ord=2)
		reward_func = lambda x: 4*np.sin(np.dot(a, x))**2
		eta = 0.01
		_lambda = 0.5
		B = 8

	bandit = ContextualBandit(time_horizon, n_arms, n_features, reward_func, noise_std=noise_std, seed=bandit_seed)

	for batch_param in [ 150, 200, 250, 300]: 

		settings = {'T': time_horizon,
					'n_arms': n_arms,
					'n_features': n_features,
					'noise std dev': noise_std,
					'nu': nu,
					'n_sim': n_sim,
					'bandit seed': bandit_seed,
					'nn seeds': nn_seeds,
					'hidden_size': hidden_size,
					'epochs': epochs,
					'batch_param': batch_param,
					'a_vec': a,
					'reward_func': fn,
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
				print('Error Occured on iteration', n, 'for', fn, 'and batch parameter', batch_param)
			regrets[n] = np.cumsum(model.regret)
			time_taken[n] = model.time_elapsed

		save_tuple = (settings, regrets, time_taken)
		filename = './' + settings['algo'] + '_' + settings['reward_func'] + '_' + str(settings['batch_param']) + '_2000_s2.pkl'
		with open(filename, 'wb') as f:
			pickle.dump(save_tuple, f)
			f.close()



import numpy as np
import itertools
import pickle
from banditreal import *
from neuralucb import *
from supnnucb import *
from newalg import *
from batchedneuralucb import *
from batchednewalg import *
from neuralts import *

# Check all settings before running

# Changes on following lines:
# Change algo name on line 63
# Change function on line 42
# Change function name on line 59
# Change B on line 33

time_horizon = 2000
n_arms = 2
n_features = 10
noise_std = 0.1
nu = 0.1
n_sim = 20
bandit_seed = 42
nn_seeds = (np.random.random(n_sim)*1000).astype('int')
delta = 0.1
eta = 0.01
_lambda = 1
lambda_0 = 1.8

# Neural Network parameters
hidden_size = 80
epochs = 200
use_cuda = False
B = 2
s = 2

# B1 = [5,10,20, 40, 80, 160,320, 640]
# B2 = [5*(2.5)**x for x in range(8)]
B2 = [10, 30, 60, 60, 60, 60, 60, 60]

filename = 'mushroom.pkl'
with open(filename, 'rb') as f:
	(X, y) = pickle.load(f)
	f.close()

X = 0.4*X
bandit = ContextualBanditReal(n_arms=n_arms, X=X, Y=y, noise_std=noise_std, seed=bandit_seed)



for batch_param in [B2]:

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
				'reward_func': 'mushroom',
				'B': B,
				'activation function': 'ReLU ' + str(s),
				'delta': delta,
				'algo': 'BatchedNewAlg_fixed',
				'eta': eta,
				'lambda': _lambda,
				'lambda_0': lambda_0 }                                        


	regrets = np.empty((n_sim, time_horizon))
	time_taken = np.empty(n_sim)


	for n in range(n_sim):
		bandit.reset_rewards()
		model = BatchedNewAlg(bandit, hidden_size=hidden_size, _lambda=_lambda, delta=delta, nu=nu, training_window=time_horizon, batch_type='fixed', lambda_0=lambda_0,
						  eta=eta, B=B, epochs=epochs, batch_param=batch_param, use_cuda=use_cuda, activation_param=s, model_seed=nn_seeds[n])#
		try:
			model.run()
		except:
			print('Error Occured on iteration', n, 'and batch parameter', batch_param)
		regrets[n] = np.cumsum(model.regret)
		time_taken[n] = model.time_elapsed

	save_tuple = (settings, regrets, time_taken)
	filename = './' + settings['algo'] + '_' + settings['reward_func'] + '_' + str(int(settings['batch_param'][1])) + '_2000_s2.pkl'
	with open(filename, 'wb') as f:
		pickle.dump(save_tuple, f)
		f.close()



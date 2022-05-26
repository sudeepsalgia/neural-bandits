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
n_sim = 30
bandit_seed = 42
nn_seeds = (np.random.random(n_sim)*1000).astype('int')
delta = 0.1
eta = 0.01
_lambda = 0.5
lambda_0 = 0.45

# Neural Network parameters
hidden_size = 80
epochs = 400
use_cuda = False
B = 2
s = 2

B1 = [3*(1**x) for x in range(8)]   #2, 2.5
B2 = [4*(1**x) for x in range(8)]     #3,4

# B1 = [300*(4**x) for x in range(8)]   #2, 2.5
# B2 = [500*(4**x) for x in range(8)]     #3,4

filename = 'shuttle.pkl'
with open(filename, 'rb') as f:
	(X, y) = pickle.load(f)
	f.close()

bandit = ContextualBanditReal(n_arms=n_arms, X=X, Y=y, noise_std=noise_std, seed=bandit_seed)


for batch_param in [B1, B2]:

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
				'algo': 'BatchedNewAlg_adaptive',
				'eta': eta,
				'lambda': _lambda,
				'lambda_0': lambda_0 }                                        


	regrets = np.empty((n_sim, time_horizon))
	time_taken = np.empty(n_sim)


	for n in range(n_sim):
		bandit.reset_rewards()
		model = BatchedNewAlg(bandit, hidden_size=hidden_size, _lambda=_lambda, delta=delta, nu=nu, training_window=time_horizon, batch_type='adaptive', lambda_0=lambda_0,
						  eta=eta, B=B, epochs=epochs, batch_param=batch_param, use_cuda=use_cuda, activation_param=s, model_seed=nn_seeds[n])
		try:
			model.run()
		except:
			print('Error Occured on iteration', n, 'and batch parameter', batch_param)
		regrets[n] = np.cumsum(model.regret)
		time_taken[n] = model.time_elapsed

	save_tuple = (settings, regrets, time_taken)
	filename = './' + settings['algo'] + '_' + settings['reward_func'] + '_' + str(int(settings['batch_param'][0]*10)) + '_2000_s2.pkl'
	with open(filename, 'wb') as f:
		pickle.dump(save_tuple, f)
		f.close()



import numpy as np
import itertools
import pickle
from banditreal import *
from linucb import *

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
reward_seeds = (np.random.random(n_sim)*1000).astype('int')
delta = 0.1
eta = 0.01   # 0.01 for mushroom, 0.001 for shuttle
_lambda = 0.5
lambda_0 = 1.8

# Neural Network parameters
hidden_size = 80
epochs = 400    # 200 for mushroom, 400 for shuttle
train_every = 5
use_cuda = False
B = 2
s = 1

filename = 'shuttle.pkl'
with open(filename, 'rb') as f:
	(X, y) = pickle.load(f)
	f.close()


bandit = ContextualBanditReal(n_arms=n_arms, X=X, Y=y, noise_std=noise_std, seed=bandit_seed)

settings = {'T': time_horizon,
			'n_arms': n_arms,
			'noise std dev': noise_std,
			'nu': nu,
			'n_sim': n_sim,
			'bandit seed': bandit_seed,
			'nn seeds': nn_seeds,
			'hidden_size': hidden_size,
			'epochs': epochs,
			'train_every': train_every,
			'reward_func': 'shuttle',
			'B': B,
			'lambda_0': lambda_0,
			'activation function': 'ReLU ' + str(s),
			'delta': delta,
			'algo': 'LinUCB',
			'eta': eta,
			'lambda': _lambda }

regrets = np.empty((n_sim, time_horizon))
time_taken = np.empty(n_sim)


for n in range(n_sim):
	bandit.reset_rewards(seed=reward_seeds[n])
	model = LinUCB(bandit, _lambda=_lambda, delta=delta, nu=nu, B=B)

	model.run()
	regrets[n] = np.cumsum(model.regret)
	time_taken[n] = model.time_elapsed

save_tuple = (settings, regrets, time_taken)
filename = './' + settings['algo'] + '_' + settings['reward_func'] + '.pkl'
with open(filename, 'wb') as f:
	pickle.dump(save_tuple, f)
f.close()



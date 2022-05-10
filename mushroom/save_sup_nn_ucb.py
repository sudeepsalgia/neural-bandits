import numpy as np
import itertools
import pickle
from bandit import *
from neuralucb import *
from supnnucb import *
from newalg import *
from neuralts import *
from batchedneuralucb import *

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
eta = 0.01
_lambda = 0.5
lambda_0 = 1.8

# Neural Network parameters
hidden_size = 50
epochs = 200
train_every = 5
use_cuda = False
B = 2
s = 1

filename = 'mushroom.pkl'
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
			'reward_func': 'mushroom',
			'B': B,
			'lambda_0': lambda_0,
			'activation function': 'ReLU ' + str(s),
			'delta': delta,
			'algo': 'SupNNUCB',
			'eta': eta,
			'lambda': _lambda }

regrets = np.empty((n_sim, time_horizon))
time_taken = np.empty(n_sim)


for n in range(n_sim):
	bandit.reset_rewards(seed=reward_seeds[n])
	model = SupNNUCB(bandit, hidden_size=hidden_size, _lambda=_lambda, delta=delta, nu=nu, training_window=time_horizon,
					  eta=eta, B=B, epochs=epochs, train_every=train_every, use_cuda=use_cuda, activation_param=s, model_seed=nn_seeds[n], lambda_0=lambda_0)

	model.run()
	regrets[n] = np.cumsum(model.regret)
	time_taken[n] = model.time_elapsed

save_tuple = (settings, regrets, time_taken)
filename = './' + settings['algo'] + '_' + settings['reward_func'] + '.pkl'
with open(filename, 'wb') as f:
	pickle.dump(save_tuple, f)
f.close()



import numpy as np
import itertools
import pickle
from bandit import *
from neuralucb import *
from supnnucb import *
from newalg import *
from batchedneuralucb import *

# Check all settings before running

# Changes on following lines:
# Change algo name on line 63
# Change function on line 42
# Change function name on line 59
# Change B on line 33

time_horizon = 1000
n_arms = 4
n_features = 10
noise_std = 0.1
nu = 0.1
n_sim = 5
bandit_seed = 42
nn_seeds = (np.random.random(10)*1000).astype('int')
delta = 0.1

# Neural Network parameters
hidden_size = 20
epochs = 200
train_every = 1
use_cuda = False
B = 8
s = 1

### mean reward function
np.random.seed(bandit_seed*2)
# a = np.random.randn(n_features)
# a /= np.linalg.norm(a, ord=2)
A = np.random.normal(scale=0.5, size=(n_features,n_features))

# 2 <a, x>^2
# reward_func = lambda x: 2*np.dot(a, x)**2
reward_func = lambda x: np.linalg.norm(np.dot(A, x), ord=2)
# reward_func = lambda x: np.cos(3*np.dot(a, x))

bandit = ContextualBandit(time_horizon, n_arms, n_features, reward_func, noise_std=noise_std, seed=bandit_seed)

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
			'train_every': train_every,
			'a_vec': A,
			'reward_func': 'xAAx',
			'B': B,
			'activation function': 'ReLU ' + str(s),
			'delta': delta,
			'algo': 'SupNNUCB'}                                        

eta_vec = [1e-3, 1e-2, 1e-1]
lambda_vec = [0.05, 0.1, 0.5]


for eta, _lambda in itertools.product(eta_vec, lambda_vec):
# eta = 0.001
# for _lambda in lambda_vec:

	regrets = np.empty((n_sim, time_horizon))
	time_taken = np.empty(n_sim)


	for n in range(n_sim):
		bandit.reset_rewards()
		model = SupNNUCB(bandit, hidden_size=hidden_size, _lambda=_lambda, delta=delta, nu=nu, training_window=time_horizon,
						  eta=eta, B=B, epochs=epochs, train_every=train_every, use_cuda=use_cuda, activation_param=s, model_seed=nn_seeds[n])

		model.run()
		regrets[n] = np.cumsum(model.regret)
		time_taken[n] = model.time_elapsed

	save_tuple = (settings, regrets, time_taken)
	filename = './Hyperparamter_' + settings['reward_func'] + '_' + settings['algo'] + '_' + str(int(-np.log10(eta))) + '_' + str(int(100*_lambda)) + '.pkl'
	with open(filename, 'wb') as f:
		pickle.dump(save_tuple, f)
	f.close()

	print(np.mean(regrets[:, -1]))

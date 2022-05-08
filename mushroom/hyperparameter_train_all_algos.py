import numpy as np
import itertools
import pickle
from banditreal import *
from neuralucb import *
from supnnucb import *
from newalg import *
# from batchedneuralucb import *
from neuralts import *

# Check all settings before running

# Changes on following lines:
# Change algo name on line 63
# Change function on line 42
# Change function name on line 59
# Change B on line 33

time_horizon = 2000
n_arms = 2
# n_features = 10
noise_std = 0.1
nu = 0
n_sim = 5
bandit_seed = 42
nn_seeds = (np.random.random(n_sim)*1000).astype('int')
delta = 0.1

# Neural Network parameters
hidden_size = 50
epochs = 200
train_every = 1
use_cuda = False
B = 0.5
s = 1

filename = 'mushroom.pkl'
with open(filename, 'rb') as f:
	(X, y) = pickle.load(f)
	f.close()

algos = ['NeuralUCB', 'SupNNUCB', 'NeuralTS', 'NewAlg']
eta_vec = [1e-3, 1e-2, 1e-1]
lambda_vec = [0.05, 0.1, 0.5]

bandit = ContextualBanditReal(n_arms=n_arms, X=X, Y=y, seed=bandit_seed)

for algo in algos:

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
				'activation function': 'ReLU ' + str(s),
				'delta': delta,
				'algo': algo}                                        


	for eta, _lambda in itertools.product(eta_vec, lambda_vec):

		regrets = np.empty((n_sim, time_horizon))
		time_taken = np.empty(n_sim)

		for n in range(n_sim):
			bandit.reset_rewards()
			if algo == 'NeuralUCB':
				model = NeuralUCB(bandit, hidden_size=hidden_size, _lambda=_lambda, delta=delta, nu=nu, training_window=time_horizon,
							  eta=eta, B=B, epochs=epochs, train_every=train_every, use_cuda=use_cuda, activation_param=s, model_seed=nn_seeds[n])
			elif algo == 'SupNNUCB':
				model = SupNNUCB(bandit, hidden_size=hidden_size, _lambda=_lambda, delta=delta, nu=nu, training_window=time_horizon,
							  eta=eta, B=B, epochs=epochs, train_every=train_every, use_cuda=use_cuda, activation_param=s, model_seed=nn_seeds[n])
			elif algo == 'NeuralTS':
				model = NeuralTS(bandit, hidden_size=hidden_size, _lambda=_lambda, delta=delta, nu=nu, training_window=time_horizon,
							  eta=eta, B=B, epochs=epochs, train_every=train_every, use_cuda=use_cuda, activation_param=s, model_seed=nn_seeds[n])
			else:
				model = NewAlg(bandit, hidden_size=hidden_size, _lambda=_lambda, delta=delta, nu=nu, training_window=time_horizon,
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



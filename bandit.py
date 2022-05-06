import numpy as np
import itertools
import torch

class ContextualBandit():
	def __init__(self, T, n_arms, n_features, h, noise_std=1.0, seed=None):

		# if not None, freeze seed for reproducibility
		self._seed(seed)

		# length of time horizon
		self.T = T 

		# total number of actions for each context
		self.n_arms = n_arms 

		# size of the feature vector for each action 
		self.n_features = n_features 

		# the reward function
		self.h = h 

		# set the standard deviation for noise which is assumed to be zero mean Gaussian
		self.noise_std = noise_std 

		# generate random features
		self.reset()

	@property
	def arms(self):
		# Return the set of arms, i.e., [0,1,2,... n_arms - 1]
		return np.arange(self.n_arms)

	def reset(self):
		# Generate new features and rewards
		self.reset_features()
		self.reset_rewards()

	def reset_features(self):
		# Generate random features on the hypersphere S^{n_features - 1}
		x = np.random.randn(self.T, self.n_arms, self.n_features)
		x /= np.repeat(np.linalg.norm(x, axis=-1, ord=2), self.n_features).reshape(self.T, self.n_arms, self.n_features)
		self.features = x

	def reset_rewards(self):
		# Generate rewards for each action at every time instant using y = h(x) + e, where e ~ Gaussian
		noisy_realizations = [self.h(self.features[t, k]) + self.noise_std*np.random.randn() for t, k in itertools.product(range(self.T), range(self.n_arms))]
		self.rewards = np.array(noisy_realizations).reshape(self.T, self.n_arms)
		
		# Find the best reward for each time instant for calculating regret
		self.best_rewards_oracle = np.max(self.rewards, axis=1)

	def _seed(self, seed=None):
		if seed is not None:
			np.random.seed(seed)
			torch.manual_seed(seed)
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)



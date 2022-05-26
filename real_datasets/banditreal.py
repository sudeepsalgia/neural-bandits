import numpy as np
import itertools
import torch

class ContextualBanditReal():
	def __init__(self, n_arms, X, Y, noise_std=0.1, seed=None):

		# if not None, freeze seed for reproducibility
		self._seed(seed)

		self.X = X
		self.Y = Y

		# length of time horizon
		self.T = len(self.Y)

		# total number of actions for each context
		self.n_arms = n_arms

		self.d = np.shape(X)[1]

		# size of the feature vector for each action 
		self.n_features = self.d*self.n_arms

		# set the standard deviation for noise which is assumed to be zero mean Gaussian
		self.noise_std = noise_std 


	@property
	def arms(self):
		# Return the set of arms, i.e., [0,1,2,... n_arms - 1]
		return np.arange(self.n_arms)

	def reset_rewards(self, seed=None):
		# Generate new features and rewards
		if seed is not None:
			np.random.seed(seed)
		new_idxs = np.random.permutation(self.T)
		x = self.X[new_idxs]
		y = self.Y[new_idxs]
		self.features = np.zeros((self.T, self.n_arms, self.n_features))
		self.rewards = np.zeros((self.T, self.n_arms))
		for t in range(self.T):
			x0 = x[t]*1.0
			# x0 /= np.linalg.norm(x0, ord=2)
			for a in range(self.n_arms):
				self.features[t][a][a*self.d:(a+1)*self.d] = x0
			self.rewards[t][y[t]] = 1

		self.rewards += np.random.normal(scale=self.noise_std, size=(self.T, self.n_arms))
		self.best_rewards_oracle = np.max(self.rewards, axis=1)

		# print(self.features[:3])

	def _seed(self, seed=None):
		if seed is not None:
			np.random.seed(seed)
			torch.manual_seed(seed)
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)



import numpy as np
from tqdm import tqdm
from utils import *
from bandit import *


class KernelUCB():

	def __init__(self, bandit, _lambda=1.0, delta=0.01, nu=-1.0, B=1, l=0.2, epochs=1, throttle=1, model_seed=42):

		# Initialize the bandit object which contains the features and the rewards
		self.bandit = bandit

		# regularization factor
		self._lambda = _lambda

		# probability of error
		self.delta = delta

		# confidence scaling factor, or equivalently the variance proxy of the sub-Gaussian noise
		if nu == -1.0:
			nu = bandit.noise_std
		self.nu = nu

		# upper bound on the hilbert space norm of the function
		self.B = B

		# kernel scaling factor
		self.l = l

		# Set the throttle for the progress bar
		self.throttle = throttle

		# reset and initialize all variables of interest to be used while the algorithm runs
		self.reset()

	@property
	def beta_t(self):
		# Calculate the beta_t factor

		return (self.B + self.nu*np.sqrt(np.sum(np.log(1 + self.samp_var[:self.iteration]/(self._lambda))) + 2 * np.log(1/self.delta)))

	def reset_UCB(self):
		# Initialize the matrices to store the posterior mean and standard deviation of all arms at all times
		self.sigma = np.empty((self.bandit.T, self.bandit.n_arms))
		self.mu = np.empty((self.bandit.T, self.bandit.n_arms))

		# Initialize a vector to store the posterior variances of the sampled points
		self.samp_var = np.empty(self.bandit.T)

		# Initialize a matrix to store all the UCBs of all arms at all times
		self.upper_confidence_bounds = np.ones((self.bandit.T, self.bandit.n_arms))

		# Set the time taken by the algorithm to run to 0
		self.time_elapsed = 0

	def reset_regret(self):
		# Initialize a vector to store all the regret at all time instants
		self.regret = np.empty(self.bandit.T)

	def reset_actions(self):
		# Initialize a vector to store the actions taken across the time horizon
		self.actions = np.empty(self.bandit.T).astype('int')

	def reset_K_inv(self):
		# Initialize the matrix that stores Z^{-1} = (lambda I + G^T G )^{-1}
		self.K_inv = 0

	def reset(self):
		# Initialize all variables of interest
		self.reset_UCB()
		self.reset_regret()
		self.reset_actions()
		self.reset_K_inv()

		# Set the iteration counter to zero
		self.iteration = 0


	def update_confidence_bounds(self):

		if self.iteration == 0:
			self.mu[self.iteration] = np.zeros(self.bandit.n_arms)
			self.sigma[self.iteration] = np.ones(self.bandit.n_arms)
		else:
			iterations_so_far = range(0, self.iteration)
			actions_so_far = self.actions[0:self.iteration]

			x_hist = self.bandit.features[iterations_so_far, actions_so_far]
			y_hist = self.bandit.rewards[iterations_so_far, actions_so_far].squeeze()

			for a in self.bandit.arms:
				x_t = self.bandit.features[self.iteration][a]
				# print(np.shape(x_hist), np.shape(x_t))
				x_diff = x_hist - x_t
				# print(np.shape(x_diff))
				k_vec = np.exp(-np.sum(x_diff**2, axis=1)/(2*(self.l**2)))
				# print(np.shape(k_vec))
				self.sigma[self.iteration][a] = np.sqrt(1 - np.dot(k_vec, np.dot(self.K_inv, np.transpose(k_vec))))
				self.mu[self.iteration][a] = np.dot(k_vec, np.dot(self.K_inv, np.transpose(y_hist)))

		# Calculate the UCBs for all actions
		self.upper_confidence_bounds[self.iteration] = self.mu[self.iteration] + self.beta_t * self.sigma[self.iteration]

	def update_K_inv(self):

		if self.iteration == 0:
			np.array([[1/(1 + self._lambda)]])
		else:
			iterations_so_far = range(0, self.iteration)
			actions_so_far = self.actions[0:self.iteration]

			x_hist = self.bandit.features[iterations_so_far, actions_so_far]

			x_diff = x_hist - self.bandit.features[self.iteration][self.action]
			k_vec = np.exp(-np.sum(x_diff**2, axis=1)/(2*(self.l**2)))
			b_K = np.array([np.dot(k_vec, self.K_inv)])
			K_22 = 1/(1 + self._lambda - np.dot(b_K, np.transpose(k_vec)))
			K_11 = self.K_inv + K_22*np.dot(np.transpose(b_K), b_K)
			K_21 = -K_22*b_K
			top_row = np.append(K_11, np.reshape(np.transpose(K_21), (self.iteration, 1)), axis=1)
			bottom_row = np.array([np.append(K_21, K_22)])
			K_inv = np.append(top_row, bottom_row, axis=0)


	def run(self):
		# Run an episode of bandit

		postfix = {'total regret': 0.0}

		with tqdm(total=self.bandit.T, postfix=postfix) as pbar:
			for t in range(1, self.bandit.T):
				# update confidence of all arms based on observed features at time t
				self.update_confidence_bounds()

				# pick action with the highest boosted estimated reward
				self.action = np.argmax(self.upper_confidence_bounds[self.iteration]).astype('int')
				self.actions[t] = self.action
				self.samp_var[t] = self.sigma[t, self.action]**2

				# update the matrix K_inv
				self.update_K_inv()

				# increment counter
				self.iteration += 1

				# compute regret
				self.regret[t] = self.bandit.best_rewards_oracle[t]-self.bandit.rewards[t, self.action]

				# log
				postfix['total regret'] += self.regret[t]

				if t % self.throttle == 0:
					pbar.set_postfix(postfix)
					pbar.update(self.throttle)

			mins, secs = pbar.format_interval(pbar.format_dict['elapsed']).split(':')
			self.time_elapsed = 60*int(mins) + int(secs)






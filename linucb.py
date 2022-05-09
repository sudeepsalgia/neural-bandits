import numpy as np
from tqdm import tqdm
from utils import *
from bandit import *


class LinUCB():

	def __init__(self, bandit, _lambda=1.0, delta=0.01, nu=-1.0, B=1, epochs=1, throttle=1):

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

		# Initialize a vector to store the estimate of theta
		self.theta = np.zeros(self.bandit.n_features)

		# Initialize a vector to store the weighted rewards
		self.b = np.zeros(self.bandit.n_features)

	def reset_regret(self):
		# Initialize a vector to store all the regret at all time instants
		self.regret = np.empty(self.bandit.T)

	def reset_actions(self):
		# Initialize a vector to store the actions taken across the time horizon
		self.actions = np.empty(self.bandit.T).astype('int')

	def reset_A_inv(self):
		# Initialize the matrix that stores Z^{-1} = (lambda I + G^T G )^{-1}
		self.A_inv = np.eye(self.bandit.n_features)/self._lambda

	def reset(self):
		# Initialize all variables of interest
		self.reset_UCB()
		self.reset_regret()
		self.reset_actions()
		self.reset_A_inv()

		# Set the iteration counter to zero
		self.iteration = 0


	def update_confidence_bounds(self):

		# Calcuating the posterior standard deviations
		self.sigma[self.iteration] = np.array([np.sqrt(np.dot(a, np.dot(self.A_inv, a.T))) for a in self.bandit.features[self.iteration]])

		# Update reward prediction mu
		self.theta = np.dot(self.A_inv, self.b)
		self.mu[self.iteration] = np.dot(self.bandit.features[self.iteration], self.theta)

		# Calculate the UCBs for all actions
		self.upper_confidence_bounds[self.iteration] = self.mu[self.iteration] + self.beta_t * self.sigma[self.iteration]

	def update_A_inv(self):
		# Update the Z_inv matrix with the action chosen at a particular time
		self.A_inv = inv_sherman_morrison(self.bandit.features[self.iteration][self.action], self.A_inv)


	def run(self):
		# Run an episode of bandit

		postfix = {'total regret': 0.0}

		with tqdm(total=self.bandit.T, postfix=postfix) as pbar:
			for t in range(self.bandit.T):
				# update confidence of all arms based on observed features at time t
				self.update_confidence_bounds()

				# pick action with the highest boosted estimated reward
				self.action = np.argmax(self.upper_confidence_bounds[self.iteration]).astype('int')
				self.actions[t] = self.action
				self.samp_var[t] = self.sigma[t, self.action]**2

				# update the matrix Z_inv
				self.update_A_inv()

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






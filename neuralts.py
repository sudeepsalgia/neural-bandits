import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import *
from bandit import *


class NeuralTS():

	def __init__(self, bandit, hidden_size=20, n_layers=2, _lambda=1.0, delta=0.01, nu=-1.0, training_window=100,
		p=0.0, eta=0.01, B=1, epochs=1, train_every=1, throttle=1,use_cuda=False, activation_param=1, model_seed=42):

		# Initialize the bandit object which contains the features and the rewards
		self.bandit = bandit

		# size of the hidden layers of the NN
		self.hidden_size = hidden_size

		# number of layers in the NN
		self.n_layers = n_layers

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

		# maximum number of rewards in the training buffer
		self.training_window = training_window

		# NN parameters
		# eta: learning rate of the NN
		# epochs: number of epochs for which to train the NN
		self.eta = eta
		self.epochs = epochs

		self.use_cuda = use_cuda
		if self.use_cuda:
			raise Exception('Not yet compatible for CUDA. Please make the necessary changes.')
		self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')

		# dropout rate of the NN
		self.p = p

		# the neural network
		self.model = Model(input_size=bandit.n_features, hidden_size=self.hidden_size, n_layers=self.n_layers, p=self.p, s=activation_param, seed=model_seed).to(self.device)
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.eta)

		# maximum L2 norm for the features across all arms and all rounds
		self.bound_features = np.max(np.linalg.norm(bandit.features, ord=2, axis=-1))

		# Set the throttle for the progress bar
		self.throttle = throttle

		# delay in feedback
		self.train_every = train_every

		# reset and initialize all variables of interest to be used while the algorithm runs
		self.reset()

	@property
	def approximator_dim(self):
		# Sum of the dimensions of all trainable layers in the network, or equivalently p

		return sum(w.numel() for w in self.model.parameters() if w.requires_grad)

	@property
	def beta_t(self):
		# Calculate the beta_t factor

		return (2*self.B + self.nu*np.sqrt(np.sum(np.log(1 + self.samp_var[:self.iteration]/(self._lambda))) + 2 * np.log(1/self.delta)))

	def reset_UCB(self):
		# Initialize the matrices to store the posterior mean and standard deviation of all arms at all times
		self.sigma = np.empty((self.bandit.T, self.bandit.n_arms))
		self.mu = np.empty((self.bandit.T, self.bandit.n_arms))

		# Initialize a vector to store the posterior variances of the sampled points
		self.samp_var = np.empty(self.bandit.T)

		# Initialize a matrix to store all the UCBs of all arms at all times
		self.ts_samples = np.ones((self.bandit.T, self.bandit.n_arms))

		# Set the time taken by the algorithm to run to 0
		self.time_elapsed = 0

	def reset_regret(self):
		# Initialize a vector to store all the regret at all time instants
		self.regret = np.empty(self.bandit.T)

	def reset_actions(self):
		# Initialize a vector to store the actions taken across the time horizon
		self.actions = np.empty(self.bandit.T).astype('int')

	def reset_Z_inv(self):
		# Initialize the matrix that stores Z^{-1} = (lambda I + G^T G )^{-1}
		self.Z_inv = np.eye(self.approximator_dim)/self._lambda

	def reset_normalized_gradient(self):
		# Initialize a matrix that stores g(a, theta)/sqrt(m) for all actions a at each time instant
		self.norm_grad = np.zeros((self.bandit.n_arms, self.approximator_dim))

	def reset(self):
		# Initialize all variables of interest
		self.reset_UCB()
		self.reset_regret()
		self.reset_actions()
		self.reset_normalized_gradient()
		self.reset_Z_inv()

		# Set the iteration counter to zero
		self.iteration = 0

	def update_output_gradient(self):
		# Get gradient of network prediction w.r.t network weights.

		for a in self.bandit.arms:
			x = torch.FloatTensor(self.bandit.features[self.iteration, a].reshape(1, -1)).to(self.device)

			self.model.zero_grad()
			y = self.model(x)
			y.backward()

			self.norm_grad[a] = torch.cat([w.grad.detach().flatten()  for w in self.model.parameters() if w.requires_grad]
				).to(self.device) #/ np.sqrt(self.hidden_size)

	def update_confidence_bounds(self):
		# Update confidence bounds and related quantities for all arms.
		self.update_output_gradient()

		# Calcuating the posterior standard deviations
		self.sigma[self.iteration] = np.array([np.sqrt(np.dot(self.norm_grad[a], np.dot(self.Z_inv, self.norm_grad[a].T))) for a in self.bandit.arms])

		# Update reward prediction mu
		self.predict()

		# Calculate the UCBs for all actions
		self.ts_samples[self.iteration] = np.random.normal(loc=self.mu[self.iteration], scale=self.beta_t * self.sigma[self.iteration])

	def update_Z_inv(self):
		# Update the Z_inv matrix with the action chosen at a particular time
		self.Z_inv = inv_sherman_morrison(self.norm_grad[self.action], self.Z_inv)

	def train(self):
		# Train the NN using the action-reward pairs in the training buffer

		iterations_so_far = range(np.max([0, self.iteration-self.training_window]), self.iteration+1)
		actions_so_far = self.actions[np.max([0, self.iteration-self.training_window]):self.iteration+1]

		x_train = torch.FloatTensor(self.bandit.features[iterations_so_far, actions_so_far]).to(self.device)
		y_train = torch.FloatTensor(self.bandit.rewards[iterations_so_far, actions_so_far]).squeeze().to(self.device)

		# train mode
		self.model.train()
		for _ in range(self.epochs):
			y_pred = self.model.forward(x_train).squeeze()
			loss = nn.MSELoss()(y_train, y_pred)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

	def predict(self):
		# Predict the output of the neural network

		# eval mode
		self.model.eval()
		self.mu[self.iteration] = self.model.forward( torch.FloatTensor(self.bandit.features[self.iteration]).to(self.device)).detach().squeeze()

	def run(self):
		# Run an episode of bandit

		postfix = {'total regret': 0.0}

		with tqdm(total=self.bandit.T, postfix=postfix) as pbar:
			for t in range(self.bandit.T):
				# update confidence of all arms based on observed features at time t
				self.update_confidence_bounds()

				# pick action with the highest boosted estimated reward
				self.action = np.argmax(self.ts_samples[self.iteration]).astype('int')
				self.actions[t] = self.action
				self.samp_var[t] = self.sigma[t, self.action]**2

				# train the nn
				if t % self.train_every == 0:
					self.train()

				# update the matrix Z_inv
				self.update_Z_inv()

				# compute regret
				self.regret[t] = self.bandit.best_rewards_oracle[t]-self.bandit.rewards[t, self.action]

				# increment counter
				self.iteration += 1

				# log
				postfix['total regret'] += self.regret[t]

				if t % self.throttle == 0:
					pbar.set_postfix(postfix)
					pbar.update(self.throttle)

			mins, secs = pbar.format_interval(pbar.format_dict['elapsed']).split(':')
			self.time_elapsed = 60*int(mins) + int(secs)






import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import *
from banditreal import *


class SupNNUCB():

	def __init__(self, bandit, hidden_size=20, n_layers=2, _lambda=1.0, delta=0.01, nu=-1.0, training_window=100, s_max=-1,
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

		# Max number of groups/models
		if s_max == -1:
			s_max = int(np.ceil(np.log(self.bandit.T)))
		self.s_max = s_max + 1 # Adding a 1 for the dummy model used to obtain gradients

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
		self.models = [Model(input_size=bandit.n_features, hidden_size=self.hidden_size, n_layers=self.n_layers, p=self.p, s=activation_param, seed=model_seed).to(self.device) for _ in range(self.s_max)]
		self.optimizers = [torch.optim.SGD(self.models[s].parameters(), lr=self.eta) for s in range(self.s_max)]

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

		return sum(w.numel() for w in self.models[-1].parameters() if w.requires_grad)

	@property
	def beta_t(self):
		# Calculate the beta_t factor

		return (self.B/np.sqrt(self._lambda) + 2*self.nu*np.sqrt(np.log(1/self.delta)))

	def reset_UCB(self):
		# Initialize the matrices to store the posterior mean and standard deviation of all arms at all times
		self.sigma = np.empty((self.bandit.T, self.bandit.n_arms))
		self.mu = np.empty((self.bandit.T, self.bandit.n_arms))

		# Initialize a vector to store the posterior variances of the sampled points
		self.samp_var = np.empty(self.bandit.T)

		# Initialize a matrix to store all the UCBs of all arms at all times
		self.upper_confidence_bounds = np.ones((self.bandit.T, self.bandit.n_arms))

	def reset_regret(self):
		# Initialize a vector to store all the regret at all time instants
		self.regret = np.empty(self.bandit.T)

	def reset_actions(self):
		# Initialize a vector to store the actions taken across the time horizon
		self.actions = np.empty(self.bandit.T).astype('int')

	def reset_Z_inv(self):
		# Initialize the matrix that stores Z^{-1} = (lambda I + G^T G )^{-1}
		self.Z_inv = [np.eye(self.approximator_dim)/self._lambda for _ in range(self.s_max)]

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

		# Set the index of the model being used to 0
		self.s = 0

		# Initialize variables to store the sampled indices for all the models
		self.iteration_idxs = [[] for _ in range(self.s_max)]
		self.action_idxs = [[] for _ in range(self.s_max)]

		# Set the time taken by the algorithm to run to 0
		self.time_elapsed = 0

	def update_output_gradient(self):
		# Get gradient of network prediction w.r.t network weights. Use the dummy model to obtain the weights

		for a in self.bandit.arms:
			x = torch.FloatTensor(self.bandit.features[self.iteration, a].reshape(1, -1)).to(self.device)

			self.models[-1].zero_grad()
			y = self.models[-1](x)
			y.backward()

			self.norm_grad[a] = torch.cat([w.grad.detach().flatten() / np.sqrt(self.hidden_size) for w in self.models[-1].parameters() if w.requires_grad]
				).to(self.device)

	def update_confidence_bounds(self):
		# Update confidence bounds and related quantities for all arms.
		self.update_output_gradient()

		# Calcuating the posterior standard deviations - Note the additional mutliplier
		self.sigma[self.iteration] =  self.beta_t * np.array([np.sqrt(np.dot(self.norm_grad[a], np.dot(self.Z_inv[self.s], self.norm_grad[a].T))) for a in self.bandit.arms])

		# Update reward prediction mu
		self.predict()

		# Calculate the UCBs for all actions
		self.upper_confidence_bounds[self.iteration] = self.mu[self.iteration] + self.sigma[self.iteration]

	def update_Z_inv(self):
		# Update the Z_inv matrix with the action chosen at a particular time
		self.Z_inv[self.s] = inv_sherman_morrison(self.norm_grad[self.action], self.Z_inv[self.s])

	def train(self):
		# Train the NN using the action-reward pairs in the training buffer
		iter_time = len(self.iteration_idxs[self.s])

		iterations_so_far = self.iteration_idxs[self.s][np.max([0, iter_time-self.training_window]):(iter_time+1)]
		actions_so_far = self.action_idxs[self.s][np.max([0, iter_time-self.training_window]):(iter_time+1)]

		x_train = torch.FloatTensor(self.bandit.features[iterations_so_far, actions_so_far]).to(self.device)
		y_train = torch.FloatTensor(self.bandit.rewards[iterations_so_far, actions_so_far]).squeeze().to(self.device)

		# train mode
		self.models[self.s].train()
		for _ in range(self.epochs):
			y_pred = self.models[self.s].forward(x_train).squeeze()
			loss = nn.MSELoss()(y_train, y_pred)
			self.optimizers[self.s].zero_grad()
			loss.backward()
			self.optimizers[self.s].step()

	def predict(self):
		# Predict the output of the neural network

		# eval mode
		self.models[self.s].eval()
		self.mu[self.iteration] = self.models[self.s].forward( torch.FloatTensor(self.bandit.features[self.iteration]).to(self.device)).detach().squeeze()

	def run(self):
		# Run an episode of bandit

		postfix = {'total regret': 0.0}
		lambda_0 = 1 #*np.sqrt(self._lambda)   # 1.8 for mushroom
		t_const = lambda_0/(self.bandit.T)**2

		with tqdm(total=self.bandit.T, postfix=postfix) as pbar:
			for t in range(self.bandit.T):
				# Initialize the set of arms in contention for that time instant
				hat_A = np.arange(self.bandit.n_arms)
				to_exit = False
				self.s = 0

				while not(to_exit):
					# update confidence of all arms based on observed features at time t
					self.update_confidence_bounds()
					
					if np.all(self.sigma[self.iteration][hat_A] <= t_const):
						self.action = hat_A[np.argmax(self.upper_confidence_bounds[self.iteration][hat_A])]
						self.regret[t] = self.bandit.best_rewards_oracle[t]-self.bandit.rewards[t, self.action]
						to_exit = True
						self.iteration += 1
					elif np.all(self.sigma[self.iteration][hat_A] <= lambda_0*2**(-(self.s+1))):
						max_LCB = np.max(self.upper_confidence_bounds[self.iteration][hat_A]) - lambda_0*2**(-self.s)
						idxs_to_keep = self.upper_confidence_bounds[self.iteration][hat_A] >= max_LCB
						if idxs_to_keep.any():
							hat_A = hat_A[idxs_to_keep]
						self.s += 1
					else:
						large_var_pts = hat_A[self.sigma[self.iteration][hat_A] > lambda_0*2**(-(self.s+1))]
						self.action = np.random.choice(large_var_pts, 1)[0]
						self.iteration_idxs[self.s].append(t)
						self.action_idxs[self.s].append(self.action)
						if len(self.iteration_idxs[self.s]) % self.train_every == 0:
							self.train()
						self.update_Z_inv()
						self.regret[t] = self.bandit.best_rewards_oracle[t]-self.bandit.rewards[t, self.action]
						self.iteration += 1
						to_exit = True
						# w[t] = self.widths[t][self.action]

				# log
				postfix['total regret'] += self.regret[t]

				if t % self.throttle == 0:
					pbar.set_postfix(postfix)
					pbar.update(self.throttle)

			mins, secs = pbar.format_interval(pbar.format_dict['elapsed']).split(':')
			self.time_elapsed = 60*int(mins) + int(secs)






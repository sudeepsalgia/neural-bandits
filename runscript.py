import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bandit import *
from neuralucb import *
from supnnucb import *
from newalg import *
from batchedneuralucb import *
from neuralts import *
from neuralgreedy import *
from linucb import *
from kernelucb import *
sns.set()

T = int(2e3)
n_arms = 4
n_features = 10
noise_std = 0.1

confidence_scaling_factor = 0.1 #noise_std

n_sim = 1

SEED = 42
np.random.seed(SEED*2)

p = 0
hidden_size = 20
epochs = 200
train_every = 1
use_cuda = False

### mean reward function
a = np.random.randn(n_features)
a /= np.linalg.norm(a, ord=2)
reward_func = lambda x: 4*np.dot(a, x)**2
# A = np.random.normal(scale=0.5, size=(n_features, n_features))
# reward_func = lambda x: np.linalg.norm(np.dot(A, x), ord=2)
# reward_func = lambda x: 4*np.sin(np.dot(a, x))**2

bandit = ContextualBandit(T, n_arms, n_features, reward_func, noise_std=noise_std, seed=SEED)


regrets = np.empty((n_sim, T))

for i in range(n_sim):
	bandit.reset_rewards()
	# model = NeuralUCB(bandit,
	# 				  hidden_size=hidden_size,
	# 				  _lambda=0.5,
	# 				  delta=0.1,
	# 				  nu=confidence_scaling_factor,
	# 				  training_window=T,
	# 				  p=p,
	# 				  eta=0.01, B=8,
	# 				  epochs=epochs,
	# 				  train_every=train_every,
	# 				  use_cuda=use_cuda,
	# 				  model_seed=100
	# 				 )

	# model = LinUCB(bandit,
	# 				  _lambda=0.5,
	# 				  delta=0.1,
	# 				  nu=confidence_scaling_factor, B=8
	# 				 )

	model = KernelUCB(bandit,
					  _lambda=0.5,
					  delta=0.1, l=0.1,
					  nu=confidence_scaling_factor, B=8
					 )

	# model = BatchedNeuralUCB(bandit,
	# 				  hidden_size=hidden_size,
	# 				  _lambda=0.5,
	# 				  delta=0.1,
	# 				  nu=confidence_scaling_factor,
	# 				  training_window=1000,
	# 				  p=p,
	# 				  eta=0.01,
	# 				  epochs=epochs,
	# 				  batch_param = 250,
	# 				  use_cuda=use_cuda
	# 				 )

	# model = NewAlg(bandit,
	# 				  hidden_size=hidden_size,
	# 				  reg_factor=1.0,
	# 				  delta=0.1,
	# 				  confidence_scaling_factor=confidence_scaling_factor,
	# 				  training_window=100,
	# 				  p=p,
	# 				  learning_rate=0.01,
	# 				  epochs=epochs,
	# 				  train_every=train_every,
	# 				  use_cuda=use_cuda
	# 				 )

	# model = SupNeuralUCB(bandit,
 #                 hidden_size=hidden_size,
 #                 reg_factor=1.0,
 #                 delta=0.1,
 #                 confidence_scaling_factor=confidence_scaling_factor,
 #                 training_window=100,
 #                 p=p,
 #                 learning_rate=0.01,
 #                 epochs=epochs,
 #                 train_every=train_every,
 #                 use_cuda=False,
 #                 )
		
	model.run()
	regrets[i] = np.cumsum(model.regret)


fig, ax = plt.subplots(figsize=(11, 4), nrows=1, ncols=1)

t = np.arange(T)

mean_regrets = np.mean(regrets, axis=0)
std_regrets = np.std(regrets, axis=0) / np.sqrt(regrets.shape[0])
ax.plot(t, mean_regrets)
ax.fill_between(t, mean_regrets - 2*std_regrets, mean_regrets + 2*std_regrets, alpha=0.15)
    
ax.set_title('Cumulative regret')

plt.tight_layout()
plt.show()

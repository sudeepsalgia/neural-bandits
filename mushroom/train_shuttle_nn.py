import numpy as np
import torch
import torch.nn as nn
import pickle 
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

filename = 'shuttle.pkl'
with open(filename, 'rb') as f:
	(X_shuttle, y_shuttle) = pickle.load(f)
	f.close()

d = np.shape(X_shuttle)[1]
n_features = 2*d
T = 2000
device = torch.device('cpu')
hidden_size = 50
epochs = 400
train_every = 10
eta = 0.01

model = Model(input_size=n_features, hidden_size=hidden_size, n_layers=2, s=1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=eta)

new_idxs = np.random.permutation(T)
X = X_shuttle[new_idxs]
y = y_shuttle[new_idxs]
features = np.zeros((T, n_features))
rewards = np.zeros(T)
for t in range(T):
	x0 = X[t]
	r = int(np.random.random() <= 0.5)
	if r == 0:
		features[t][:d] = x0
	else:
		features[t][d:] = x0
	if r == y[t]:
		rewards[t] = 1


	# features[2*t][:d] = x0
	# features[2*t+1][d:] = x0
	# rewards[2*t + y[t]] = 1

X_train, X_test, Y_train, Y_test = train_test_split(features, rewards, test_size = 0.2)

x_train = torch.FloatTensor(X_train).to(device)
y_train = torch.FloatTensor(Y_train).squeeze().to(device)

loss_vec = np.zeros(epochs)

# train mode
model.train()
for i in range(epochs):
	y_pred = model.forward(x_train).squeeze()
	loss = nn.MSELoss()(y_train, y_pred)
	loss_vec[i] = loss.detach().item()
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

model.eval()
model_preds = model.forward(torch.FloatTensor(X_test).to(device)).squeeze()
print(nn.MSELoss()(torch.FloatTensor(Y_test).squeeze().to(device), model_preds).detach().item())

fig, ax = plt.subplots(figsize=(11, 4), nrows=1, ncols=1)

ep = np.arange(epochs)

ax.plot(ep, loss_vec)
 
plt.tight_layout()
plt.show()






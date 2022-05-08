import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('./datasets/data_banknote_authentication.txt', sep=',', header=None)
df.columns = ['var', 'skew', 'kurt', 'ent', 'class']



X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# labelencoder = LabelEncoder()
# # mm = MinMaxScaler()
sc = StandardScaler()

# y = labelencoder.fit_transform(y)
X = sc.fit_transform(X)

filename = './mushroom/bank.pkl'
with open(filename, 'wb') as f:
	pickle.dump((X, y), f)
f.close()


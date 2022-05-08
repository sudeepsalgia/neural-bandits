import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('./datasets/winequality-red.csv')

df = df[df['quality'] > 4]
quals = df['quality']
quals[quals > 7] = 7
df['quality'] = quals

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

labelencoder = LabelEncoder()
sc = StandardScaler()

y = labelencoder.fit_transform(y)
X = sc.fit_transform(X)

filename = 'wine.pkl'
with open(filename, 'wb') as f:
	pickle.dump((X, y), f)
f.close()


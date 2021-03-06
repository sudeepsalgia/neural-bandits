import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('./datasets/winequality-red.csv')

df = df[df['quality'] > 4]
df = df[df['quality'] < 7]
# quals = df['quality']
# quals[quals > 6] = 6
# df['quality'] = quals

# print(df.groupby("quality").size())

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

labelencoder = LabelEncoder()
mm = MinMaxScaler()
sc = StandardScaler()

y = labelencoder.fit_transform(y)
X = mm.fit_transform(X)

filename = './mushroom/wine.pkl'
with open(filename, 'wb') as f:
	pickle.dump((X, y), f)
f.close()


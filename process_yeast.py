import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_fwf('./datasets/yeast.data')

df = df[df['class'].isin(['MIT', 'CYT', 'NUC', 'ME3'])]
# print(df.head())

# df = df[df['quality'] > 4]
# df = df[df['quality'] < 7]

X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

labelencoder = LabelEncoder()
# mm = MinMaxScaler()
sc = StandardScaler()

y = labelencoder.fit_transform(y)
X = sc.fit_transform(X)

filename = './mushroom/yeast.pkl'
with open(filename, 'wb') as f:
	pickle.dump((X, y), f)
f.close()


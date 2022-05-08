import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle
from scipy import stats

df = pd.read_csv('./datasets/telescope_data.csv')
# print(df['class'].sample(n=10))

z = np.abs(stats.zscore(df.iloc[:, 1:-1]))

df = df[(z < 3).all(axis=1)]

df1 = df[df['class'] == 'g'].sample(n=1000)
df0 = df[df['class'] == 'h'].sample(n=1000)

dataset = pd.concat([df0, df1])

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

labelencoder = LabelEncoder()
sc = StandardScaler()

y = labelencoder.fit_transform(y)
X = sc.fit_transform(X)

filename = './mushroom/magic.pkl'
with open(filename, 'wb') as f:
	pickle.dump((X, y), f)
f.close()


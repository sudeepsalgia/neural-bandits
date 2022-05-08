import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle
import scipy.io as sio


mat_file = sio.loadmat('./datasets/shuttle.mat')

X = mat_file['X']
y = mat_file['y']

col_names = [str(x) for x in range(9)]

dict_file = {col_names[i]: X[:, i] for i in range(9)}
dict_file['labels'] = np.squeeze(y)

df = pd.DataFrame.from_dict(dict_file)

df1 = df[df['labels'] == 0].sample(n=1000)
df0 = df[df['labels'] == 1].sample(n=1000)

dataset = pd.concat([df0, df1])

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

sc = StandardScaler()
X = sc.fit_transform(X)

filename = 'shuttle.pkl'
with open(filename, 'wb') as f:
	pickle.dump((X, y), f)
f.close()
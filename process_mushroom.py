import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle


df = pd.read_csv('./datasets/mushrooms.csv')
df = df.astype('category')

labelencoder=LabelEncoder()
for column in df.columns:
	df[column] = labelencoder.fit_transform(df[column])

df = df.drop(["veil-type"],axis=1)
df.reset_index(drop=True, inplace=True)

df1 = df[df['class'] == 1].sample(n=1000)
df0 = df[df['class'] == 0].sample(n=1000)

df_new = pd.concat([df0, df1])

ct = df_new.to_numpy()

X_mushroom = ct[:, 1:]
y_mushroom = ct[:, 0]

filename = 'mushroom.pkl'
with open(filename, 'wb') as f:
	pickle.dump((X_mushroom, y_mushroom), f)
f.close()



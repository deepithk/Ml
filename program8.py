#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:12:25 2020

@author: ubuntu
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler as SS
from sklearn.mixture import GaussianMixture as GM
#import sklearn.metrics as sm
import pandas as pd
import numpy as np


iris_dataset = pd.read_csv('iris.csv')
iris_dataset['Targets'] = iris_dataset.Class.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

X = iris_dataset[['Sepal_Length','Sepal_Width','Petal_Length', 'Petal_Width']]

Y = iris_dataset[['Targets']]

model = KMeans(n_clusters = 3, random_state=0)
model.fit(X)
print('Model Labels: \n', model.labels_)
scaler = SS()
scaler.fit(X)

xs = scaler.transform(X)
gmm = GM(n_components = 3, random_state=0)
gmm.fit(xs)

Y_gmm = gmm.predict(xs)
print('GMM Labels: \n',Y_gmm)

plt.figure(figsize = (14,14))
colormap = np.array(['red','lime','black']) 

plt.subplot(2,2,1)
plt.scatter(X.Petal_Length, X.Petal_Width, c= colormap[Y.Targets], s=40)
plt.title('Real Classification')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')



plt.subplot(2,2,2)
plt.scatter(X.Petal_Length, X.Petal_Width, c= colormap[model.labels_], s=40)
plt.title('K Means Clustering')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')


plt.subplot(2,2,3)
plt.scatter(X.Petal_Length, X.Petal_Width, c= colormap[Y_gmm], s=40)
plt.title('GMM Based Clustering')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')


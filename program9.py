#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 11:11:18 2020

@author: ubuntu
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split as tts
import pandas as pd

dataset = pd.read_csv("iris.csv")
X = dataset[['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']]
Y = dataset['Class']

x_train,x_test,y_train,y_test = tts(X,Y,random_state=0,test_size=0.25)
classifier = KNeighborsClassifier(n_neighbors=8, p=3, metric = 'euclidean')
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is as follows: \n', cm)
print('Accuracy Metrics:')
print(classification_report(y_test,y_pred))
print("Correct Prediction: ",accuracy_score(y_test,y_pred))
print("Wrong Prediction: ",(1-accuracy_score(y_test,y_pred)))

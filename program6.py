#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:30:03 2020

@author: ubuntu
"""

import pandas as pd
txt = pd.read_csv("Text.csv",names=['text','label'])
print(txt)
print("\n Total instances in the dataset is: ",txt.shape[0])

txt['labelnum'] = txt.label.map({'pos':1, 'neg': 0})

X = txt.text
Y = txt.labelnum

from sklearn.model_selection import train_test_split as tts
xtrain, xtest, ytrain, ytest = tts(X,Y,random_state=0)

print("\nDataset is split into Training and Testing Samples")
print("Total training instances:",xtrain.shape[0])
print(xtrain)
print("Total training instances:",xtest.shape[0])
print(xtest)

#Output of count vectorizer is a sparse matrix
#CountVectorizer - stands for "feature extraction" 

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm = count_vect.transform(xtest)

print("\nTotal features extracted using CountVectorizer",xtrain_dtm.shape[1])
print("\nFeatures for training instances are:")
df = pd.DataFrame(xtrain_dtm.toarray(),columns=count_vect.get_feature_names())
print(df.columns)
print("\nDocument term matrix is: \n")
print(df)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_dtm,ytrain)
predicted = clf.predict(xtest_dtm)

print("\n---Classification results of testing samples are given below---")
for doc,p in zip(xtest,predicted):
    pred = "pos" if p==1 else "neg"
    print("%s -> %s"%(doc,pred))
    
from sklearn import metrics
print("\nAccuracy of the classifier is: ",metrics.accuracy_score(ytest,predicted))
print("Recall of the classifier is: ",metrics.recall_score(ytest,predicted))
print("Precision of the classifier is: ",metrics.precision_score(ytest,predicted))
print("Confusion matrix is: ")
print(metrics.confusion_matrix(ytest,predicted))

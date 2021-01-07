#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:19:46 2020

@author: ubuntu
"""

import pandas as pd
import numpy as np

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

attributes = ['age', 'sex', 'cp' ,'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']

heartDisease = pd.read_csv('HeartDisease.csv', names=attributes)
heartDisease = heartDisease.replace('?', np.nan)

print('Few examples from the dataset are given below')
print(heartDisease.head())
print('\nAttributes and datatypes')
print(heartDisease.dtypes)

model = BayesianModel( [ ('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('sex', 'trestbps'),
('exang', 'trestbps'), ('trestbps', 'heartdisease'), ('fbs', 'heartdisease'), ('heartdisease', 'restecg'),
('heartdisease', 'thalach'), ('heartdisease', 'chol') ] )

print('\nLearning Conditional Probability Distributions using Maximum Likelihood Estimators...')

model.fit(heartDisease, estimator = MaximumLikelihoodEstimator)
print('\nInferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)

print('\nComputing the probability of Heart disease given age = 28')
q = HeartDisease_infer.query(variables = ['heartdisease'], evidence = {'age':28})

print(q['heartdisease'])
print('\nComputing the probability of Heart disease given chol = 100')

q = HeartDisease_infer.query(variables=['heartdisease'], evidence = {'chol':100})
print(q['heartdisease'])
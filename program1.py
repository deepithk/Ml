#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:25:50 2020

@author: ubuntu
"""

import csv
attributes = [['Sunny','Cloudy','Rainy'],['Warm','Cold'],['Low','Normal','High'],['Strong','Weak'],['Warm','Cool'],['Same','Change']]

total_attributes = len(attributes)

print("The total number of attributes is: ",total_attributes)
print("The most specific hypothesis is: ",['0','0','0','0','0','0'])
print("The most general hypothesis is: ",['?','?','?','?','?','?'])
a = []
print("The given training dataset is: \n")

cfile = open('EnjoySport.csv','r')
for row in csv.reader(cfile):
    a.append(row)
    print(row)

print("The total number of records: ",len(a))
print("The initial hypothesis: \n")
hypothesis = ['0']*total_attributes
print(hypothesis)

for i in range(0,len(a)):
    if a[i][total_attributes]=='Yes':
        for j in range(0,total_attributes):
            if hypothesis[j] == '0' or hypothesis[j] == a[i][j]:
                hypothesis[j] = a[i][j]
            else:
                hypothesis[j] = '?'
            
    print("\nHypothesis for Training Example Number {} is \n".format(i+1),hypothesis)

print("\nThe Maximally Specific Hypothesis for a given training examples: \n",hypothesis)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:12:04 2019

@author: milind
"""

# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
from sklearn.model_selection import train_test_split
import operator

#%%

fileName = "Batch08.csv"
df = pd.read_csv(fileName)

cols = df.columns.tolist()
cols.remove("Class")

X = df[cols]
Y = df["Class"]

X_data, Y_data, X_label, Y_label = train_test_split(X, Y, test_size=0.33, random_state=42)



#%%

# Feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)

f = dict(zip(cols, fit.scores_))

score_x = sorted(f.items(), key=operator.itemgetter(1), reverse = True)

names, scores = zip(*score_x)
curr_col = [];
for i in range(len(names)):
    curr_col.append(names[i])
    X[curr_col].to_csv("best_{0}_features".format(i+1), index = False)
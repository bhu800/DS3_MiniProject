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
from scipy.stats import chi2_contingency

#%%

fileName = "Batch08.csv"
df = pd.read_csv(fileName)

cols = df.columns.tolist()
cols.remove("Class")

X = df[cols]
Y = df["Class"]

X_data, Y_data, X_label, Y_label = train_test_split(X, Y, test_size=0.33, random_state=42)



#%%
#dependent = 1
#independent = 0
def chi_test(c, X, Y, alpha = 0.05):
    table = pd.crosstab(X, Y)
    stat, p, dof, expected = chi2_contingency(table)
    if p <= alpha:
        print(c,p,"Dependent", sep='\t')
        return 1
    else:
        print(c,p,"Independent", sep = '\t')
        return 0

for c in cols:
    chi_test(c, X[c], Y)











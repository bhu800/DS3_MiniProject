#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 02:13:21 2019

@author: milind
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../Data-Preprocessing/Batch08.csv')
data = df.loc[:, df.columns != "Class"]

corr = data.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),square=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');
plt.savefig("corr matrix.png")

for a in data.columns.tolist():
    print("**********************************************")
    print(a)
    print(corr[a])
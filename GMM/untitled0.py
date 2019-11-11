#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:48:53 2019

@author: milind
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import os

coot = '/home/milind/Documents/projects/DS3_MiniProject/Data-Preprocessing'
imgF = '/home/milind/Documents/projects/DS3_MiniProject/GMM/images'

#%%
def LogisticReger(fileName):
	# fileName = "Batch08.csv"
	dirn = os.path.basename(os.path.dirname(fileName))
	df = pd.read_csv(fileName)
	fnm = os.path.split(fileName)

	cols = df.columns.tolist()
	cols.remove("Class")

	X = df[cols]
	Y = df["Class"]

	X_data, Y_data, X_label, Y_label = train_test_split(X, Y, test_size=0.33, random_state=42)

	logisticRegr = LogisticRegression()
	logisticRegr.fit(X_data, X_label)
	#score = logisticRegr.score(Y_data, Y_label)
	predictions = logisticRegr.predict(Y_data)
	accScore = metrics.accuracy_score(predictions, Y_label)
	confMatrix = metrics.confusion_matrix(predictions, Y_label)
	print(accScore)
	print(confMatrix)

	cm = confMatrix
	score = accScore
	plt.figure(figsize=(9,9))
	sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
	plt.ylabel('Actual label');
	plt.xlabel('Predicted label');
	all_sample_title = 'File : {0}\n\n'.format(fnm[1])
	all_sample_title += 'Accuracy Score: {0}'.format(score)
	plt.title(all_sample_title, size = 15)
#	plt.xticks([1, 2])
#	plt.yticks([1, 2])
	plt.savefig(dirn + "_{0}.png".format(fnm[1]))

#%%
fileList = []
for root, dirs, files in os.walk(coot):
     for file in files:
        fpath = os.path.join(root, file)
        if(fpath.find('.csv') != -1):
            fileList.append(fpath)

for fi in fileList:
    LogisticReger(fi)

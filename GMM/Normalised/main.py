import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix

os.chdir("../../Data-Preprocessing/Normalised")
for csv_file in os.listdir():
    if csv_file[-4:] != ".csv":
        continue
    data = data = pd.read_csv(csv_file)
    try:
        del data["Unnamed: 0"]
    except:
        pass
    data_train, data_test = train_test_split(data, test_size=0.30)
    X_test, y_test = data_test.values[:, :-1], data_test.values[:, -1]
    grp1 = data_train.groupby("Class").get_group(1).values[:, :-1]
    grp2 = data_train.groupby("Class").get_group(2).values[:, :-1]
    ACC = []
    Q = []
    for q in range(1, 10):
        gmm1 = GaussianMixture(n_components=q)
        gmm1.fit(grp1)
        gmm2 = GaussianMixture(n_components=q)
        gmm2.fit(grp2)
        prior1 = len(grp1) / len(data_train)
        prior2 = len(grp2) / len(data_train)
        score_arr1 = gmm1.score_samples(X_test) * prior1
        score_arr2 = gmm2.score_samples(X_test) * prior2
        y_pred = []
        for i in range(len(score_arr1)):
            if score_arr1[i] > score_arr2[i]:
                y_pred.append(1)
            else:
                y_pred.append(2)
        y_pred = np.array(y_pred)
        acc = accuracy_score(y_pred, y_test)
        ACC.append(acc)
        Q.append(q)
    plt.plot(Q, ACC)
    plt.xlabel("Number of Components")
    plt.ylabel("Accuracy")
    plt.savefig(
        "C:/Users/kuldip/Desktop/DS3Project/DS3_MiniProject/GMM/Normalised/"
        + csv_file[:-4]
    )
    plt.show()
    Q_max_acc = np.argmax(ACC) + 1
    max_acc = max(ACC)
    print(csv_file[:-4], max_acc, Q_max_acc)


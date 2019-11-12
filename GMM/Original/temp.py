import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = open("Original.txt", "r")
l1 = file.readlines()
file.close()
l = []
for i in range(1, len(l1)):
    try:
        try:
            a = int(l1[i].split(" ")[0][17:19])
        except:
            a = int(l1[i].split(" ")[0][17:18])
        b = float(l1[i].split(" ")[1])
        l.append((a, b))
    except:
        pass
l.sort()
X = []
Y = []
for i in l:
    X.append(i[0])
    Y.append(i[1])
plt.plot(X, Y)
plt.xlabel("Number of Dimensions")
plt.ylabel("Accuracy")
plt.title("Analysis_of_Original_Data")
plt.savefig("Analysis_of_Original_Data")
plt.show()

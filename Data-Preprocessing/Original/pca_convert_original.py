from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("original_data.csv")
data = df.iloc[:, 0:15]

data.columns = np.arange(0, 15)
# cols.remove('REFILL_COUNT')

data.boxplot()
print(df.columns)
plt.xlabel("Attributes Indices")
plt.title("Orginal Data")
plt.savefig("orig.png")
plt.show()

pca = PCA(n_components=15)
pca.fit(data)
ll = pca.explained_variance_
xx = np.arange(1, 16)

plt.plot(xx, ll)

plt.title("Orginal Data")
plt.ylabel("Eigen Values")
plt.xlabel("Eigen Vectors")
plt.savefig('eig.png')
plt.show()

print(ll)
print(pd.DataFrame(np.array(data.iloc[:, 0:10].corr())))
# print(data.head())

# %%
for i in range(1, 16, 1):
    pca = PCA(n_components=i)
    pca.fit(data)
    x = pca.transform(data)
    pca_df = pd.DataFrame(x)
    converted_df = pd.concat([pca_df, df['Class']], axis=1)

    str1 = str("pca_df_reduced_to"+str(i)+"dimensions"+".csv")

    converted_df.to_csv(str1, index=False)

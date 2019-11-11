from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("normalised_data.csv")
data=df.iloc[:,0:15]


pca=PCA(n_components=15)
pca.fit(data)
ll=pca.explained_variance_
xx=np.arange(1,16)

plt.plot(xx,ll)
plt.show()

pd.DataFrame(np.array(data.iloc[:,].cov()))

plt.savefig('eig.png')

#%%
for i in range(1,16,1):    
    pca=PCA(n_components=i)
    pca.fit(data)
    x=pca.transform(data)
    pca_df=pd.DataFrame(x)
    converted_df=pd.concat([pca_df,df['Class']],axis=1)
    str1=str("pca_df_reduced_to"+str(i)+"dimensions"+".csv")
    converted_df.to_csv(str1)
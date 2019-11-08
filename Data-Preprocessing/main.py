import pandas as pd
import os

df=pd.read_csv('Batch08.csv')

data=df.iloc[:,0:15]

col=df.columns.tolist()
#col.remove('Class')

ROOT_DIR=os.getcwd()
print(ROOT_DIR)
#%%

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
x=scaler.fit_transform(data)

new_x=pd.DataFrame(x)
standardised_df=pd.concat([new_x,df['Class']],axis=1)

standardised_df.columns=col

standardised_dir=os.path.join(ROOT_DIR,'Standardised')
standardised_df.to_csv(os.path.join(standardised_dir,'standardised_data.csv'))

#%%


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
x=scaler.fit_transform(data)

new_x=pd.DataFrame(x)
normalised_df=pd.concat([new_x,df['Class']],axis=1)

normalised_df.columns=col

normalised_dir=os.path.join(ROOT_DIR,'Normalised')
normalised_df.to_csv(os.path.join(normalised_dir,'normalised_data.csv'))




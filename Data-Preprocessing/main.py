import pandas as pd
import os

#Imported the original csv file as pandas dataframe
df=pd.read_csv('Batch08.csv')

#Selecting first 15 columns of the dataframe excluding Column "Class"
data=df.iloc[:,0:15]

#Assigning all the column names to the list col
col=df.columns.tolist()
#col.remove('Class')

#getting the root directory of the main.py file
ROOT_DIR=os.getcwd()
print(ROOT_DIR)

#%%

#Saving the original csv file to origial data directory 
original_data_dir=os.path.join(ROOT_DIR,'Original')

df.to_csv(os.path.join(original_data_dir,'original_data.csv'),index=False)


#%%

from sklearn.preprocessing import StandardScaler

#Imported StandarScaler and Standardised the data
scaler=StandardScaler()

#saved it to x
x=scaler.fit_transform(data)

#converted to a new dataframe
new_x=pd.DataFrame(x)

#concated it with the Class data and stored it in standardisd df
standardised_df=pd.concat([new_x,df['Class']],axis=1)

standardised_df.columns=col

#saved it to the Standardised directory
standardised_dir=os.path.join(ROOT_DIR,'Standardised')
standardised_df.to_csv(os.path.join(standardised_dir,'standardised_data.csv'),index=False)

#%%


from sklearn.preprocessing import MinMaxScaler

#Imported MinMaxScaler and Normalised the data
scaler=MinMaxScaler()

#saved it to x
x=scaler.fit_transform(data)

#converted to a new dataframe
new_x=pd.DataFrame(x)

#concated it with the Class data and stored it in normalised df
normalised_df=pd.concat([new_x,df['Class']],axis=1)

normalised_df.columns=col

#saved it to the Normalised directory
normalised_dir=os.path.join(ROOT_DIR,'Normalised')
normalised_df.to_csv(os.path.join(normalised_dir,'normalised_data.csv'),index=False)




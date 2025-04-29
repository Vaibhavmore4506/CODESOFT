import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler

df=pd.read_csv("Titanic.csv")

df=df.iloc[:,[0,4,5,9]]

df['Fare']=df['Fare'].fillna(df['Fare'].mean())
df['Age']=df['Age'].fillna(df['Age'].median())
df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])

df.to_csv('rawDataset.csv')
le=LabelEncoder()
df['Gender']=le.fit_transform(df[['Age']])
scaler=MinMaxScaler()
df['Age']=scaler.fit_transform(df[['Age']])
df['Fare']=scaler.fit_transform(df[['Fare']])
df['Age']=df['Age'].round(2)
df['Fare']=df['Fare'].round(2)
df.to_csv('preProcessed.csv')
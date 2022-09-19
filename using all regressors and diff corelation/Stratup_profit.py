import pandas as pd
data=pd.read_csv("F:/Techinest/AI/Classes/Aspire---18 april/Regressions_with_multiple algo/50_Startups.csv")
print(data.isna().sum())

from sklearn.impute import SimpleImputer
SI=SimpleImputer(strategy='mean')
SI=SI.fit(data[['R&D Spend','Administration','Marketing Spend']])
data[['R&D Spend','Administration','Marketing Spend']]=SI.transform(data[['R&D Spend','Administration','Marketing Spend']])
SI_State=SimpleImputer(strategy='most_frequent')
SI_State=SI_State.fit(data[['State']])
data['State']=SI_State.transform(data[['State']])
print(data.isna().sum())

SI_R__D=SI.fit(data[['R&D Spend']])
data['R&D Spend']=SI_R__D.transform(data[['R&D Spend']])


SI_Administration=SI.fit(data[['Administration']])
data['Administration']=SI_Administration.transform(data[['Administration']])


SI_Marketing_Spend=SI.fit(data[['Marketing Spend']])
data['Marketing Spend']=SI_Marketing_Spend.transform(data[['Marketing Spend']])


SI_State=SimpleImputer(strategy='most_frequent')
SI_State=SI_State.fit(data[['State']])
data['State']=SI_State.transform(data[['State']])
print(data.isna().sum())

from sklearn.preprocessing import LabelEncoder
LB=LabelEncoder()
data['State']=LB.fit_transform(data['State'])

print(data.corr()['Profit'])
data=data.drop('State',axis=1)

import matplotlib.pyplot as plt
plt.plot(data['R&D Spend'],data['Profit'])
plt.xlabel("R&D spend")
plt.ylabel("Profit spend")
plt.show()

plt.plot(data['Profit'],data['Marketing Spend'])
plt.xlabel('Profit spend')
plt.ylabel("Marketing Spend")
plt.show()

X=data.iloc[:,0:3].values
Y=data.iloc[:,-1].values


#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
#ct=ColumnTransformer([('OHE',OneHotEncoder(drop='first'),['State'])],remainder='passthrough')
#data=ct.fit_transform(data)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split  
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=2)

#
#
#
#
#

y_pred=model.predict(X_test)
from sklearn.metric import r2_score
score=r2_score(y_test,y_pred)

import pandas as pd
data=pd.read_csv("C:/Users/Subham/Desktop/Learn N Build/Machine learning algorithm by kris sir/Assignments to do/IRIS.csv")
x=data['petal_length']
a=x.mean()
print(a)
data_greater_than_mean=x[x>a]
print(data_greater_than_mean)

print(data.isna().sum())
print(data.head())


'''from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
CT=ColumnTransformer([('OHE',OneHotEncoder(drop='first'),['species'])],remainder='passthrough')
data=CT.fit_transform(data)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)'''

x=data.iloc[:,0:4]
y=data.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=.3,random_state=2)

from sklearn.svm import SVC
model=SVC(kernel='rbf',random_state=0)
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
score=accuracy_score(Y_test,y_pred)
print(score)

import matplotlib.pyplot as plt
plt.scatter(Y_test,y_pred,color='r')
plt.show()
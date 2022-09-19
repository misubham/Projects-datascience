import pandas as pd
data=pd.read_csv("C:/Users/Subham/Desktop/Learn N Build/Machine learning algorithm by kris sir/Random forest(RF)/Social_Network_Ads.csv")
x=data.iloc[:,1:4].values
y=data.iloc[:,-1].values
y=y.reshape(y.shape[0],1)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
CT=ColumnTransformer([('OHE',OneHotEncoder(drop='first'),[0])],remainder='passthrough')
x=CT.fit_transform(x)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc=sc.fit(x)
x=sc.transform(x)
#x=sc.fit_transform(x)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=.2,random_state=23)

from sklearn.ensemble import RandomForestClassifier as r
accuracy=[]
for i in range (1,15):
    model=r(n_estimators=i,random_state=2)
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    from sklearn.metrics import accuracy_score
    score=accuracy_score(Y_pred,Y_test)
    accuracy.append(score)
    
import matplotlib.pyplot as plt
x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]
plt.plot(x,accuracy,color='black')
plt.title('nestimators curve with accuracy')
plt.xlabel('n_estimators value')
plt.ylabel('accuracy of model')
plt.show()

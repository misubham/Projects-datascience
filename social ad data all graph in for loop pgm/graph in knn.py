import pandas as pd
data=pd.read_csv("C:/Users/Subham/Desktop/Learn N Build/Machine learning algorithm by kris sir/Assignments to do/social ad data all graph in for loop pgm/Social_Network_Ads.csv")
print(data.isna().sum())

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
CT=ColumnTransformer([('OHE',OneHotEncoder(drop='first'),['Gender'])],remainder='passthrough')
data=CT.fit_transform(data)

X=data[:,1:4]
Y=data[:,-1]
Y=Y.reshape(Y.shape[0],1)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=23)

from sklearn.neighbors import KNeighborsClassifier
accuracy=[]
for i in range(1,10):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    from sklearn.metrics import accuracy_score
    score=accuracy_score(Y_test,Y_pred)
    accuracy.append(score)
    
import matplotlib.pyplot as plt
x=[1,2,3,4,5,6,7,8,9]
plt.plot(x,accuracy,color='red')
plt.title('accuracy curve to neighbor value change')
plt.xlabel('neighbors value')
plt.ylabel('accuracy score')
plt.show()

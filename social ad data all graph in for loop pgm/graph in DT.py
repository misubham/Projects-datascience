import pandas as pd
data=pd.read_csv("C:/Users/Subham/Desktop/Learn N Build/Machine learning algorithm by kris sir/Decession tree(DT)/Social_Network_Ads.csv")
#print(data.isna().sum())
#print(data.head())

X=data.iloc[:,1:4].values
Y=data.iloc[:,-1].values
Y=Y.reshape(Y.shape[0],1)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
trans=ColumnTransformer([('Tf1',OneHotEncoder(drop='first'),[0])],remainder='passthrough')
X=trans.fit_transform(X)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=23)

from sklearn.tree import DecisionTreeClassifier
accuracy=[]
for i in range(1,10):
    model=DecisionTreeClassifier(max_depth=i)
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    from sklearn.metrics import accuracy_score
    score=accuracy_score(Y_pred,Y_test)
    accuracy.append(score)
print(accuracy)
x=[1,2,3,4,5,6,7,8,9]
import matplotlib.pyplot as plt
plt.plot(x,accuracy,color='g')
plt.title('accuray graph to depth increases')
plt.xlabel('max depth of DT')
plt.ylabel('accuracy of our model')
plt.show()


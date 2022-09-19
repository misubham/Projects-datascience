import pandas as pd
data=pd.read_csv("C:/Users/Subham/Desktop/Learn N Build/Aspire Assignment and Project(submitted)/Assignments of ML/Heart data set/heart.csv")
print(data.isna().sum())
print(data.head(1))

X_corr=data.corr()
print(X_corr)

x=data.iloc[:,[2,5,7,8,9,10,11,12]].values
y=data.iloc[:,-1].values
y=y.reshape(y.shape[0],1)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc=sc.fit(x)
x=sc.transform(x)
#x=sc.fit_transform(x)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=.2,random_state=23)

from sklearn.ensemble import RandomForestClassifier as r
accuracy=[]
for i in range(1,13):
    model=r(n_estimators=i,random_state=2)
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(Y_pred,Y_test)
    from sklearn.metrics import accuracy_score
    score=accuracy_score(Y_pred,Y_test)
    accuracy.append(score)
    print(accuracy)

import matplotlib.pyplot as plt
x=[1,2,3,4,5,6,7,8,9,10,11,12]
plt.plot(x,accuracy,color='black')
plt.title('nestimators curve with accuracy')
plt.xlabel('n_estimators value')
plt.ylabel('accuracy of model')
plt.show()

'''import pickle
heart="C:/Users/Subham/Desktop/Learn N Build/Machine learning algorithm by kris sir/Assignments to do/Heart data set/heart.sav"
file=open('C:/Users/Subham/Desktop/Learn N Build/Machine learning algorithm by kris sir/Assignments to do/Heart data set/heart.sav','w+')
file.write("This is a test file")
file.close()
pickle.dump(model,open(heart,'wb'))'''

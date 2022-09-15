import pandas as pd
train=pd.read_json("C:/Users/Subham/Downloads/train.json/train.json")
print(train.isna().sum())
X=train.iloc[:,[0,2]]
Y=train.iloc[:,[1]]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=2)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(max_depth=5)
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

import pickle
Pur_model_modified="C:/Users/Subham/Desktop/pur_model_modi.sav"
pickle.dump(model,open(Pur_model_modified,'wb'))


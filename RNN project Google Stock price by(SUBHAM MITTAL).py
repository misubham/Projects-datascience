#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os


# In[5]:


dataset_train = pd.read_csv("C:/Users/Subham/Downloads/trainset.csv")


# In[6]:


dataset_train


# In[13]:


trainset = dataset_train.iloc[:,1:2].values


# In[14]:


trainset


# In[15]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_scaled = sc.fit_transform(trainset)


# In[16]:


training_scaled


# In[17]:


x_train = []
y_train = []


# In[18]:


for i in range(60,1259):
    x_train.append(training_scaled[i-60:i, 0])
    y_train.append(training_scaled[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)


# In[19]:


x_train.shape


# In[20]:


x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


# In[21]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[22]:


regressor = Sequential()
regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))


# In[23]:


regressor.add(Dropout(0.2))


# In[24]:


regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))


# In[25]:


regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))


# In[26]:


regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


# In[27]:


regressor.add(Dense(units = 1))


# In[28]:


regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')


# In[57]:


regressor.fit(x_train,y_train,epochs =20, batch_size = 32)


# In[58]:


dataset_test =pd.read_csv("C:/Users/Subham/Downloads/testset.csv")


# In[59]:


real_stock_price = dataset_test.iloc[:,1:2].values


# In[60]:


dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0)
dataset_total


# In[61]:


inputs = dataset_total[len(dataset_total) - len(dataset_test)-60:].values
inputs


# In[62]:


inputs = inputs.reshape(-1,1)


# In[49]:


inputs


# In[63]:


inputs = sc.transform(inputs)
inputs.shape


# In[64]:


x_test = []
for i in range(60,185):
    x_test.append(inputs[i-60:i,0])


# In[65]:


x_test = np.array(x_test)
x_test.shape


# In[67]:


x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
x_test.shape


# In[68]:


predicted_price = regressor.predict(x_test)


# In[69]:


predicted_price = sc.inverse_transform(predicted_price)
predicted_price


# In[70]:


plt.plot(real_stock_price,color = 'red', label = 'Real Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


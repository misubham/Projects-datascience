#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

import cv2 as cv
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


# In[2]:


Class_name=os.listdir("C:/Users/Subham/Downloads/train")


# In[3]:


Class_name


# In[10]:


plt.figure(figsize=(15,11))
path = f"C:/Users/Subham/Downloads/train/{Class_name[0]}"
print(f"                         ********************************{Class_name[0]}*************************")
for i in range(1,7):
    plt.subplot(3,3,i)
    plt.tight_layout()
    rand_img = cv.imread(path +'/'+ np.random.choice(sorted(os.listdir(path))))
    plt.imshow(rand_img)
    plt.xticks([])
    plt.yticks([])


# In[9]:


plt.figure(figsize=(15,11))
path = f"C:/Users/Subham/Downloads/train/{Class_name[1]}"
print(f"                         ********************************{Class_name[1]}*************************")
for i in range(1,7):
    plt.subplot(3,3,i)
    plt.tight_layout()
    rand_img = cv.imread(path +'/'+ np.random.choice(sorted(os.listdir(path))))
    plt.imshow(rand_img)
    plt.xticks([])
    plt.yticks([])


# In[10]:


plt.figure(figsize=(15,11))
path = f"C:/Users/Subham/Downloads/train/{Class_name[2]}"
print(f"                         ********************************{Class_name[2]}*************************")
for i in range(1,7):
    plt.subplot(3,3,i)
    plt.tight_layout()
    rand_img = cv.imread(path +'/'+ np.random.choice(sorted(os.listdir(path))))
    plt.imshow(rand_img)
    plt.xticks([])
    plt.yticks([])


# In[11]:


plt.figure(figsize=(15,11))
path = f"C:/Users/Subham/Downloads/train/{Class_name[3]}"
print(f"                         ********************************{Class_name[3]}*************************")
for i in range(1,7):
    plt.subplot(3,3,i)
    plt.tight_layout()
    rand_img = cv.imread(path +'/'+ np.random.choice(sorted(os.listdir(path))))
    plt.imshow(rand_img)
    plt.xticks([])
    plt.yticks([])


# In[12]:


# Plotting 12 images to check dataset
plt.figure(figsize=(15,11))
path = f"C:/Users/Subham/Downloads/train/{Class_name[4]}"
print(f"                         ********************************{Class_name[4]}*************************")
for i in range(1,7):
    plt.subplot(3,3,i)
    plt.tight_layout()
    rand_img = cv.imread(path +'/'+ np.random.choice(sorted(os.listdir(path))))
    plt.imshow(rand_img)
    plt.xticks([])
    plt.yticks([])


# In[13]:


# Plotting 12 images to check dataset
plt.figure(figsize=(15,11))
path = f"C:/Users/Subham/Downloads/train/{Class_name[5]}"
print(f"                         ********************************{Class_name[5]}*************************")
for i in range(1,7):
    plt.subplot(3,3,i)
    plt.tight_layout()
    rand_img = cv.imread(path +'/'+ np.random.choice(sorted(os.listdir(path))))
    plt.imshow(rand_img)
    plt.xticks([])
    plt.yticks([])


# In[14]:


# Plotting 12 images to check dataset
plt.figure(figsize=(15,11))
path = f"C:/Users/Subham/Downloads/train/{Class_name[6]}"
print(f"                         ********************************{Class_name[6]}*************************")
for i in range(1,7):
    plt.subplot(3,3,i)
    plt.tight_layout()
    rand_img = cv.imread(path +'/'+ np.random.choice(sorted(os.listdir(path))))
    plt.imshow(rand_img)
    plt.xticks([])
    plt.yticks([])


# In[4]:


train_count=[]
for i in Class_name:
    train_count.append(len(os.listdir("C:/Users/Subham/Downloads/train/"+i+"/")))
plt.bar(Class_name,train_count)


# In[5]:


train_datagen = ImageDataGenerator(
                                         width_shift_range = 0.1,
                                         height_shift_range = 0.1,
                                         horizontal_flip = True,
                                         rescale = 1./255,
                                         zoom_range = 0.2,
                                         validation_split = 0.2
                                        
                                        )
test_datagen = ImageDataGenerator(rescale = 1./255,
                                        )
train_generator = train_datagen.flow_from_directory(directory = "C:/Users/Subham/Downloads/train/",
                                                    target_size = (48,48),
                                                    batch_size = 32,
                                                    color_mode = "grayscale",
                                                    class_mode = "categorical"
                                                   )
test_generator = test_datagen.flow_from_directory( directory = "C:/Users/Subham/Downloads/test",
                                                              target_size = (48,48),
                                                              batch_size = 32,
                                                              color_mode = "grayscale",
                                                              class_mode = "categorical",
                                                             )


# In[6]:


CNN= Sequential()
CNN.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(48,48,1)))
CNN.add(Conv2D(64,(3,3),padding='same',activation='relu'))
CNN.add(BatchNormalization())
CNN.add(MaxPool2D(pool_size=(2, 2)))
CNN.add(Dropout(0.25))

CNN.add(Conv2D(128,(3,3),padding='same',activation='relu'))
CNN.add(Conv2D(256,(3,3),padding='same',activation='relu'))
CNN.add(BatchNormalization())
CNN.add(MaxPool2D(pool_size=(2, 2)))
CNN.add(Dropout(0.25))

CNN.add(Conv2D(256,(3,3),padding='same',activation='relu'))
CNN.add(BatchNormalization())
CNN.add(MaxPool2D(pool_size=(2, 2)))
CNN.add(Dropout(0.25))

CNN.add(Flatten()) 

CNN.add(Dense(100,activation = 'relu'))

CNN.add(Dense(50,activation = 'relu'))

CNN.add(Dense(225,activation = 'relu'))

CNN.add(BatchNormalization())
CNN.add(Dropout(0.25))

CNN.add(Dense(7, activation='softmax'))


# In[7]:


CNN.summary()


# In[8]:


CNN.compile(
    optimizer = tf.keras.optimizers.Adam(lr=0.0005), 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])


# In[9]:


history = CNN.fit(
    train_generator ,
    epochs=2)


# In[11]:


CNN.evaluate(test_generator)


# In[12]:


train_acc = history.history['accuracy']
train_loss = history.history['loss']

plt.xkcd()
plt.style.use('seaborn-deep')

plt.rcParams["figure.figsize"] = (20, 10) 

plt.plot(history.history['accuracy'],marker="*",label="Acc Value")
plt.plot(history.history['loss'],marker="*",label="loss Value")
plt.grid(True)

plt.title("this is a good graph")
plt.xlabel("Epoch Number")
plt.ylabel("")

plt.show()


# In[ ]:





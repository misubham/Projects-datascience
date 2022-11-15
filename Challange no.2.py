#!/usr/bin/env python
# coding: utf-8

# # challange no 2 
# # subham mittal

# ## In an array N1-N2 numbers are stored, one number is missing, find the missing number. For example N1=1 and N2=10 1,2,3,4,6,7,8,9,10. Answer -> 5

# In[8]:


arr=[1,2,3,4,6,7,8,9,10,11,14]
lst=[]
for i in range (arr[0],arr[-1]+1):
    if i not in arr:
        lst.append(i)
print(lst)


# ## In an array 1-100 multiple numbers are duplicates, Find all the duplicate numbers

# In[1]:


arr=[1,2,3,3,4,4,5,5 ,6,7,8,9,9,77,77,88,89]
for i in range(0,len(arr)):
    for j in range(i+1,len(arr)):
        if(arr[i]==arr[j]):
            print(arr[j])


# In[ ]:





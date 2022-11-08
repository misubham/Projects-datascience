#!/usr/bin/env python
# coding: utf-8

# # Challange no.1(learn and build)
# ## by subham mittal

# In[42]:


def poss_combination(string,i=0):
    if i==len(string): 
        print("".join(string))
    for j in range(i,len(string)):
        words = [c for c in string]
   # swaping the combinations
        words[i],words[j]=words[j],words[i]
        poss_combination(words,i+1)

print("printing the possible combination of string '123'\n ")
print(poss_combination("123"))


# In[58]:


#Method creation of removing b letter
class subham_remove:
    def __init__(self,subh_method):
        self.subh_method=subh_method
        print(subh_method.replace("l",""))


# In[59]:


b=subham_remove("Hello Learn and Build")


# In[60]:


dir(b)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[5]:


cancer=load_breast_cancer()


# In[7]:


cancer.keys()


# In[9]:


print(cancer["DESCR"])


# In[10]:


df_feat=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[11]:


df_feat.head(2)


# In[12]:


df_feat.info()


# In[13]:


cancer['target']


# In[14]:


cancer['target_names']


# In[16]:


from sklearn.model_selection import train_test_split


# In[23]:


X=df_feat
y=(cancer['target'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[20]:


from sklearn.svm import SVC


# In[21]:


model=SVC()


# In[25]:


model.fit(X_train,y_train)


# In[26]:


predictions=model.predict(X_test)


# In[27]:


from sklearn.metrics import classification_report, confusion_matrix


# In[28]:


print(classification_report(y_test,predictions))


# In[29]:


print(confusion_matrix(y_test,predictions))


# In[33]:


from sklearn.model_selection import GridSearchCV


# In[34]:


param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}


# In[35]:


grid= GridSearchCV(SVC(),param_grid,verbose=3)


# In[36]:


grid.fit(X_train,y_train)


# In[37]:


grid.best_params_


# In[38]:


grid.best_estimator_


# In[39]:


grid.best_score_


# In[42]:


grid_predictions=grid.predict(X_test)


# In[43]:


print(classification_report(y_test,grid_predictions))


# In[44]:


print(confusion_matrix(y_test,grid_predictions))


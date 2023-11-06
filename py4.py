#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn import preprocessing


# In[2]:


df = pd.read_csv('diabetes.csv')


# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


df.corr().style.background_gradient(cmap='BuGn')


# In[6]:


df.drop(['BloodPressure', 'SkinThickness'], axis=1, inplace=True)


# In[7]:


df.isna().sum()


# In[8]:


X=df.iloc[:, :df.shape[1]-1]       #Independent Variables
y=df.iloc[:, -1]                   #Dependent Variable
X.shape, y.shape


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[24]:


knn = KNeighborsClassifier(n_neighbors = 27)


# In[25]:


knn.fit(X_train, y_train)


# In[26]:


y_pred = knn.predict(X_test)


# In[27]:


ConfusionMatrixDisplay.from_predictions(y_test, y_pred)


# In[28]:


print(classification_report(y_test, y_pred))


# In[38]:


from sklearn import metrics
knnn=KNeighborsClassifier(n_neighbors=27)
y_pred=knnn.fit(X_train, y_train).predict(X_test)
print(f"Accuracy for KNN model \t: {metrics.accuracy_score(y_test, y_pred)}")


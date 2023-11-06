#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('sales_data_sample.csv', encoding='latin1')
df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df= df[['PRICEEACH', 'MSRP']]


# In[7]:


df.head()


# In[8]:


df.isna().any()


# In[9]:


df.describe().T


# In[10]:


df.shape


# In[15]:


from sklearn.cluster import KMeans

inertia = []

for i in range(1, 11):
    clusters = KMeans(n_clusters=i, init='k-means++', random_state=42)
    clusters.fit(df)
    inertia.append(clusters.inertia_)
    
plt.figure(figsize=(6, 6))
sns.lineplot(x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], y = inertia)


# In[16]:


kmeans.cluster_centers_


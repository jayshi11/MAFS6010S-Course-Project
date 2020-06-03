#!/usr/bin/env python
# coding: utf-8

# In[253]:


import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# In[327]:


# Load data
train_data = pd.read_csv('train.csv', dtype = {'target' : np.uint8})

test_data = pd.read_csv('test.csv')

song = pd.read_csv('songs.csv')

member = pd.read_csv('members.csv', dtype={'bd' : np.uint8})

extra = pd.read_csv('song_extra_info.csv')


# In[328]:


train_data.head()


# In[329]:


test_data.head()


# In[330]:


song.head()


# In[331]:


member.head()


# In[332]:


extra.head()


# In[333]:


# Data virtulization
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
train_data.isnull().sum().plot(kind='bar')
plt.title('Train set Missing Value Count')
plt.subplot(1, 2, 2)
test_data.isnull().sum().plot(kind='bar')
plt.title('Test set Missing Value Count')


# In[334]:


plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
song.isnull().sum().plot(kind='bar')
plt.title('Song Missing Value Count')
plt.subplot(1, 2, 2)
member.isnull().sum().plot(kind='bar')
plt.title('Member Missing Value Count')


# In[335]:


plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
extra.isnull().sum().plot(kind='bar')
plt.title('Extra info Missing Value Count')


# In[336]:


# Data preprocessing
fig, axs = plt.subplots(figsize = (10,4), ncols=2)
x = np.array(member.bd)
sns.boxplot(x,ax=axs[0])
axs[0].set_title('member age boxplot')
axs[0].set(xlabel='Year old')
sns.distplot(x,ax=axs[1])
axs[1].set_title('member age distribution')
axs[1].set(xlabel='Year old')


# In[337]:


# Replace outlier age with medium age
median = member.loc[(member['bd'])<100 & (member['bd']>0), 'bd'].median()
member.loc[(member.bd > 100), 'bd'] = np.nan
member.loc[(member.bd == 0), 'bd'] = np.nan
member.bd.fillna(median,inplace=True)


# In[338]:


# New age plots
fig, axs = plt.subplots(figsize = (10,4), ncols=2)
x = np.array(member.bd)
sns.boxplot(x,ax=axs[0])
axs[0].set_title('member age boxplot')
axs[0].set(xlabel='Year old')
sns.distplot(x,ax=axs[1])
axs[1].set_title('member age distribution')
axs[1].set(xlabel='Year old')


# In[339]:


# registered methods
fig, axs = plt.subplots(figsize = (10,4), ncols=2)
x = np.array(member.registered_via)
sns.boxplot(x,ax=axs[0])
axs[0].set_title('registered methods boxplot')
axs[0].set(xlabel='Index')
sns.distplot(x,ax=axs[1])
axs[1].set_title('registered methods distribution')
axs[1].set(xlabel='Index')


# In[340]:


# registration time
fig, axs = plt.subplots(figsize = (10,4), ncols=2)
x = np.array(member.registration_init_time/1e4)
sns.boxplot(x,ax=axs[0])
axs[0].set_title('registration time boxplot')
axs[0].set(xlabel='Time')
sns.distplot(x,ax=axs[1])
axs[1].set_title('registration time distribution')
axs[1].set(xlabel='Time')


# In[341]:


# expiration time
fig, axs = plt.subplots(figsize = (10,4), ncols=2)
x = np.array(member.expiration_date/1e4)
sns.boxplot(x,ax=axs[0])
axs[0].set_title('expiration date boxplot')
axs[0].set(xlabel='Time')
sns.distplot(x,ax=axs[1])
axs[1].set_title('expiration date distribution')
axs[1].set(xlabel='Time')


# In[342]:


# replace outlier with its corresponding registration time
member.loc[(member.expiration_date < member.registration_init_time), 'expiration_date'] = member.registration_init_time+1


# In[343]:


# expiration time processed
fig, axs = plt.subplots(figsize = (10,4), ncols=2)
x = np.array(member.expiration_date/1e4)
sns.boxplot(x,ax=axs[0])
axs[0].set_title('expiration date boxplot')
axs[0].set(xlabel='Time')
sns.distplot(x,ax=axs[1])
axs[1].set_title('expiration date distribution')
axs[1].set(xlabel='Time')


# In[344]:


# Song length
fig, axs = plt.subplots(figsize = (10,4), ncols=2)
x = np.array(song.song_length/(1000))
sns.boxplot(x,ax=axs[0])
axs[0].set_title('song length boxplot')
axs[0].set(xlabel='Second')
sns.distplot(x,ax=axs[1])
axs[1].set_title('song length distribution')
axs[1].set(xlabel='Second')


# In[345]:


# merge extra song info into train and test datasets
train = train_data
test = test_data
train = train.merge(extra, on='song_id', how='left')
test = test.merge(extra, on='song_id', how='left')


# In[346]:


# merge member info into train and test datasets
train = train.merge(member, on='msno', how='left')
test = test.merge(member, on='msno', how='left')


# In[347]:


# merge song info into train and test sets
train = train.merge(song, on='song_id', how='left')
test = test.merge(song, on='song_id', how='left')


# In[348]:


# Convert object type to category, fill NA value
for col in train.drop('target', axis = 1).columns:
    if train[col].dtype == 'object':
        train[col] = train[col].fillna('unknown') # fill na as unknown
        test[col] = test[col].fillna('unknown')
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
    elif col == 'bd' or col == 'song_length':
        train[col] = train[col].fillna(train[col].median())
        test[col] = test[col].fillna(test[col].median())
    else:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)


# In[349]:


train.dtypes


# In[350]:


train.isnull().sum()


# In[351]:


# Prepare train sets
X = train.drop('target', axis = 1)
y = train.target.values


# In[370]:


get_ipython().run_cell_magic('time', '', "# lgbm training\nX_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8, random_state=1234)\n\nlgbm = lgb.LGBMClassifier(application = 'binary',\n                          metric = 'auc',\n                          learning_rate = 0.2,\n                          max_depth = 8,\n                          num_leaves = 2**8,\n                          num_iterations = 2000)\nlgbm.fit(X_train, y_train, eval_set=(X_validation, y_validation), verbose = 100)\n\n#save model\nfilename = 'lgbm.sav'\njoblib.dump(lgbm, filename)")


# In[365]:


X_test = test.drop('id', axis = 1)
y_test = lgbm.predict_proba(X_test)[:,-1]


# In[366]:


result_lgbm = pd.DataFrame()
result_lgbm['id'] = test['id'].values


# In[367]:


result_lgbm['target'] = y_test


# In[368]:


result_lgbm.to_csv('submission_lgbm_V4.csv', index= False, float_format = '%.5f')


# In[243]:


get_ipython().run_cell_magic('time', '', "# catboost training\n\nX_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8, random_state=1234)\n\ncat_features_names = [col for col in X.columns if X[col].dtype.name == 'category']\ncat_features = [X.columns.get_loc(col) for col in cat_features_names]\nSEED = 1\nparams = {'loss_function':'Logloss',\n          'iterations' : 2000,\n          'eval_metric':'AUC',\n          'boosting_type' : 'Plain',\n          'leaf_estimation_iterations' : 1,\n          #'task_type' : 'GPU',\n          'cat_features': cat_features,\n          'early_stopping_rounds': 100,\n          'verbose': True,\n          'random_seed': SEED\n         }\ncb = CatBoostClassifier(**params)\n\ncb.fit(X_train, y_train, \n       eval_set=(X_validation, y_validation),\n       plot=True)\n\n#save model\nfilename = 'catboost.sav'\njoblib.dump(cb, filename)")


# In[244]:


X_test = test.drop('id', axis = 1)
y_test = cb.predict_proba(X_test)[:,-1]

len(y_test)


# In[245]:


result_cb = pd.DataFrame()
result_cb['id'] = test['id'].values


# In[246]:


result_cb['target'] = y_test

result_cb.to_csv('submission_cb.csv', index= False, float_format = '%.5f')


# In[371]:


# Eval Virtualization
plt.figure(figsize=(10,4))
lgb.plot_metric(lgbm)


# In[372]:


plt.figure(figsize=(10,4))
lgb.plot_importance(lgbm)


# In[377]:


lgb.plot_tree(lgbm,figsize=(100, 100))


# In[380]:


fea_imp = pd.DataFrame({'imp': cb.feature_importances_, 'col': X.columns})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
fea_imp.plot(kind='barh', x='col', y='imp', figsize=(6, 4), legend=None)
plt.title('CatBoost - Feature Importance')
plt.ylabel('Features')
plt.xlabel('Importance');


# In[383]:


cb.plot_tree(tree_idx=1999, pool=None)


# In[ ]:





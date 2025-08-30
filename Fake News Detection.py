#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re    #regular expression
import string


# In[3]:


data_fake = pd.read_csv('fake.csv')
data_true = pd.read_csv('true.csv')


# In[4]:


data_fake.head()


# In[5]:


data_true.head()


# In[6]:


data_fake["class"]=0


# In[7]:


data_true['class']=1


# In[8]:


data_fake.shape, data_true.shape


# In[9]:


data_fake_manual_testing = data_fake.tail(10)
for i in range(23480,23470,-1):
    data_fake.drop([i], axis=0, inplace = True)


# In[10]:


data_true_manual_testing = data_true.tail(10)
for i in range(21416,21406,-1):
    data_true.drop([i], axis=0, inplace = True)


# In[11]:


data_fake.shape, data_true.shape


# In[12]:


data_fake_manual_testing['class']=0
data_true_manual_testing['class']=1


# In[13]:


data_fake_manual_testing.head(10)


# In[14]:


data_true_manual_testing.head(10)


# In[15]:


data_merge = pd.concat([data_fake, data_true], axis=0)
data_merge.head(10)


# In[16]:


data_merge.columns


# Removing those columns which are not required

# In[17]:


data = data_merge.drop(['title','subject','date'], axis = 1)


# In[18]:


data.isnull().sum()      #null value checking in the datasets


# In[19]:


data = data.sample(frac = 1)   


# In[20]:


data.head()


# In[21]:


data.reset_index(inplace = True)
data.drop(['index'], axis = 1, inplace =True)


# In[22]:


data.columns


# In[23]:


data.head()


# Creating the function to process the text in the datasets

# In[24]:


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# In[25]:


data['text'] = data['text'].apply(wordopt)


# In[26]:


x = data['text']
y = data['class']


# In[27]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)


# In[28]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# In[29]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xv_train, y_train)


# In[30]:


pred_lr = LR.predict(xv_test)


# In[31]:


LR.score(xv_test, y_test)


# In[32]:


print(classification_report(y_test, pred_lr))


# Random forest decision tree classifier

# In[33]:


from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[34]:


pred_dt = DT.predict(xv_test)


# In[35]:


DT.score(xv_test, y_test)


# In[36]:


print(classification_report(y_test, pred_lr))


# Gradiant Boosting Classifier Technique

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train, y_train)


# In[ ]:


pred_gb = GB.predict(xv_test)


# In[ ]:


GB.score(xv_test, y_test)


# In[41]:


print(classification_report(y_test, pred_gb))


# In[42]:


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state = 0)
RF.fit(xv_train, y_train)


# In[43]:


pred_rf = RF.predict(xv_test)


# In[44]:


RF.score(xv_test, y_test)


# In[45]:


print(classification_report(y_test, pred_rf))


# In[52]:


def output_lable(n):
    if n==0:
        return "Fake News"
    elif n==1:
        return "Not a fake news"
    
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    
    return print("\n\nLR prediction: {} \nDT predictin: {} \nGB prediction{} \nRF prediction{}".format(output_lable(pred_LR[0]), output_lable(pred_GB[0]), output_lable(pred_RF[0])))


# In[53]:


news = str(input())
manual_testing(news)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





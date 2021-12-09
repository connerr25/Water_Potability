#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.style.use('dark_background')
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tqdm import tqdm_notebook
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from luciferml.preprocessing import Preprocess as prep


# In[2]:


data = pd.read_csv('water_potability.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.shape


# In[7]:


data.isnull().sum()


# In[8]:


data = data.dropna()


# In[9]:


data.isnull().sum()


# In[10]:


plt.figure(figsize=(10,6)) #setting the figure size
sns.countplot(data['Potability'], palette='rocket') # checking the class count of potable water
plt.title('Potability count', weight='bold')
plt.tight_layout()
#total potability count plotted


# In[11]:


non_potable = data[data['Potability']==0]
percent_non_potable = len(non_potable)/ len(data)
print('The percentage of non potable water is: {}%'.format(round(percent_non_potable * 100,4)))


# In[12]:


data.nunique()


# In[13]:


colors = sns.color_palette('twilight')[0:6]
sns.palplot(colors)
#set up color palette for boxen plot


# In[14]:


#Boxen Plot of each Column except Solids
df1 = pd.DataFrame()
df1 = data
df1 = df1.drop("Solids",1)
fig1, ax = plt.subplots(figsize=[20,10])
ax = sns.boxenplot(data=df1, orient="h", palette=colors)
sns.despine(offset=10, trim=True)
plt.title("Boxen Plot of each Column except Solids", fontsize=20);
plt.show()


# In[15]:


columns = data[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
       'Organic_carbon', 'Trihalomethanes', 'Turbidity']]


# In[16]:


columns.shape


# In[17]:


def distributions(data):
    
    
    features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
       'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    
    plt.figure(figsize=(16,16))
    for i in tqdm_notebook(range(len(data.columns)), desc = 'loading'):
        plt.subplot(3,3,i+1)
        sns.distplot(data[data.columns[i]], color='red', rug=True)
        plt.title(data.columns[i], weight='bold')
        plt.tight_layout()
        
distributions(columns)


# In[18]:


def pairplt(data):
    
    sns.pairplot(data, hue='Potability', palette='OrRd')
    plt.tight_layout()
    
pairplt(data)


# In[19]:


data.hist(bins=20, color = 'green', figsize=(16,16))
plt.tight_layout()


# In[20]:


data.columns


# In[21]:


def attributes_and_potability(data):
 
    #getting count of everything with respect to potability
    features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
       'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    for var in tqdm_notebook(features, desc = 'loading'):
        plt.figure(figsize=(16,10))
        sns.histplot(data = data, x = data[var], hue ='Potability') # histogram
        plt.title(var) # title of the plot
        
attributes_and_potability(data)


# In[22]:


plt.figure(figsize=(16,12))
matrix = np.triu(data.corr()) # matrix to return k -th diagonal zeroed values.
sns.heatmap(data.corr(), annot=True, mask=matrix, cmap='OrRd') # creating correlational map
plt.title('Correlational Map', weight='bold');


# In[ ]:





# In[ ]:





# In[23]:


X = data.iloc[:,:-1].values
y = data.iloc[:,-1:].values

X.shape, y.shape


# In[24]:


scaler = StandardScaler()
X = scaler.fit_transform(X)
X


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0) # data split 85-15


# In[26]:


X_train, y_train = shuffle(X_train, y_train) # data shuffling


# In[27]:


y_train.shape, y_test.shape


# In[28]:


pipeline = make_pipeline(RobustScaler()) # creating a pipeline for all the models
Random_forest = make_pipeline(pipeline, RandomForestClassifier(random_state=0, min_samples_leaf = 2, n_estimators = 1000))
Decision_tree = make_pipeline(pipeline, DecisionTreeClassifier(random_state=0))
Logistic_regression = make_pipeline(pipeline, LogisticRegression(random_state=0))
svc = make_pipeline(pipeline, SVC(random_state=0))
KNeighbors = make_pipeline(pipeline, KNeighborsClassifier())
Ada_boost = make_pipeline(pipeline, AdaBoostClassifier(random_state=0))
xgboost = make_pipeline(pipeline, XGBClassifier())
gradientboost = make_pipeline(pipeline, GradientBoostingClassifier(random_state=0))


# In[29]:


param_dist = {
    'RandomForest':Random_forest,
    'DecisionTree':Decision_tree,
    'LogisticRegression':Logistic_regression,
    'svc':svc,
    'KNeighbors':KNeighbors,
    'AdaBoost':Ada_boost,
    'XGB':xgboost,
    'GD':gradientboost
}


# In[30]:


def MODEL(model):
    
    #run model and generate confusion matrix 
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('The accuracy score of the model is: {}%'.format(accuracy_score(y_test, y_pred)))
    print('-'*50)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# In[31]:


def model_evaluation(parameter_dictionary):
    
  #  calculate precision, recall, f1 score
    
    for name, model in parameter_dictionary.items():
        print('-'*50)
        print(name)
        evaluation = MODEL(model)
    return evaluation
evaluation = model_evaluation(param_dist)


# In[32]:


accuracy_score_model = {
    'RandomForest':71.9,
    'DecisionTree':61.6,
    'LogisticRegression':63,
    'svc':73.6,
    'KNeighbors':65,
    'AdaBoost':60,
    'XGB':66,
    'GD':68
    
}


# In[33]:


def models_overview(accuracy_score_model):
    
   
    #compare models to each other
    
    model_accuracy = list(accuracy_score_model.values())
    model_name = list(accuracy_score_model.keys())

    g = sns.barplot(x = model_accuracy, y = model_name,palette='OrRd')
    plt.title('Models Overview', weight='bold');
    return g
    
over_view = models_overview(accuracy_score_model)


# In[34]:


svc = SVC(random_state=0)
svc.fit(X_train, y_train)


# In[35]:


y_pred = svc.predict(X_test)


# In[36]:


def svc_report(y_test, y_pred, X_test, svc):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True)
    plot_roc_curve(svc, X_test, y_test)
    print(classification_report(y_test, y_pred))
    
svc_report(y_test, y_pred, X_test, svc)


# In[37]:


print('The accuracy score of the model is: {}% '.format(round(accuracy_score(y_test, y_pred)*100, 2)))


# In[38]:


#SVC is best model


# In[ ]:





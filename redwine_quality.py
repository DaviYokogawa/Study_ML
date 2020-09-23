#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

######### Doing Descriptive Analisys #########

df = pd.read_csv('winequality-red.csv') #Importing dataset
df.head()
df.quality = df.quality.astype(float) # Transforming dtype of quality column
df.info() # Taking Info about data
df.quality.describe() # Describing quality column
quality_bool = [] # Creating a bool object

# Loop for to append values to quality_bool
for i in df.quality.iteritems():
    value = (i[1])
    if value >= 6.5:
        quality_bool.append(1)
    else:
        quality_bool.append(0)

df['quality_bool'] = quality_bool # Putting the Column in Data Frame
df.quality_bool.value_counts() # Understanding the comportament of my data

#Importing ProfileReport to help on Descriptive Analisys
from pandas_profiling import ProfileReport

profile = ProfileReport(df, title='Relatory of Red Wine Quality', html={'style':{'full_width':True}})
profile
profile.to_file(output_file="redwine_quality.html") #Dowloading the relatory

# Cleaning Data
df.duplicated() # Finding duplicated rows
df_without_duplicates = df.drop_duplicates() # Creating othes Data Frame withour duplicated rows

# Comparing Data Frames
df.info()
df_without_duplicates.info()
df_without_duplicates.mean()
df_without_duplicates = df_without_duplicates.rename(columns={'citric acid': 'citric_acid'})
df_without_duplicates.head()
df_without_duplicates['citric_acid'].mean()
df_without_duplicates['citric_acid'][df_without_duplicates.citric_acid > 0].mean()
df_without_duplicates.loc[df_without_duplicates.citric_acid == 0, 'citric_acid'] = 0.29535787321063395
df_without_duplicates.head()

# Plotting some graphics to understand the data

fig = plt.figure(figsize = (10,5))
sns.barplot(x = 'quality', y = 'fixed acidity', data = df_without_duplicates)
# IRRELEVANT

fig = plt.figure(figsize = (10,5))
sns.barplot(x = 'quality', y = 'volatile acidity', data = df_without_duplicates)
# THE LESS VOLATILE ACIDITY BETTER

fig = plt.figure(figsize = (10,5))
sns.barplot(x = 'quality', y = 'citric_acid', data = df_without_duplicates)
# THE MORE CITRIC ACID BETTER

fig = plt.figure(figsize = (10,5))
sns.barplot(x = 'quality', y = 'residual sugar', data = df_without_duplicates)
# IRRELEVANT

fig = plt.figure(figsize = (10,5))
sns.barplot(x = 'quality', y = 'chlorides', data = df_without_duplicates)
# THE LESS CHLORIDES BETTER

fig = plt.figure(figsize = (10,5))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = df_without_duplicates)
# FOLLOW THE NORMAL CURVE

fig = plt.figure(figsize = (10,5))
sns.barplot(x = 'quality', y = 'density', data = df_without_duplicates)
# IRRELEVANT

fig = plt.figure(figsize = (10,5))
sns.barplot(x = 'quality', y = 'pH', data = df_without_duplicates)
# IRRELEVANT

fig = plt.figure(figsize = (10,5))
sns.barplot(x = 'quality', y = 'sulphates', data = df_without_duplicates)
# THE MORE SULFATE BETTER

fig = plt.figure(figsize = (10,5))
sns.barplot(x = 'quality', y = 'alcohol', data = df_without_duplicates)
# THE MORE ALCOHOL BETTER

######### Machine Lerning #########

df_without_duplicates.head()

# Deleting the columns that I will not use
X = df_without_duplicates.drop(['fixed acidity', 'residual sugar', 'density', 'pH', 'quality', 'quality_bool'], axis = 1)
X.head()
y = df_without_duplicates['quality_bool'] # Defining my y axis
y.head()

from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
######### Creating a DummyClassier to use like a baseline #########
dummy = DummyClassifier(strategy='uniform', random_state=123)
dummy.fit(X_train, y_train)
dummy_baseline = dummy.score(X_test, y_test) # Our baseline

sc = StandardScaler() # Creating the variable to scale our data

######### Scaling and fitting our data #########

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

######### Testing Models #########

# Testing Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier( random_state = 123)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))

rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)
rfc_eval_percent = rfc_eval.mean()*100
print(f"Random Forest Classifier without Tunning: {rfc_eval_percent}")

# Testing Support Vector Machine

from sklearn.svm import SVC

svc = SVC(random_state = 123)
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)

print(classification_report(y_test, pred_svc))
print(confusion_matrix(y_test, pred_svc))

svc_eval = cross_val_score(estimator = svc, X = X_train, y = y_train, cv = 10)
svc_eval_percent = svc_eval.mean()*100
print(f"Support Vector Machine without Tunning: {svc_eval_percent}")

######### Tunning Models #########

# Tunning RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score

param = { 
    'n_estimators': [100, 500],
    'criterion' :['gini', 'entropy'],
    'max_depth' : [4,5,6,7,8],
    'max_features': ['auto', 'sqrt', 'log2'],
}
grid_rfc = GridSearchCV(estimator = rfc, param_grid = param, cv = 10) # Creating the variable to grid our model
grid_rfc.fit(X_train, y_train)
grid_rfc.best_params_

rfc2 = RandomForestClassifier(n_estimators = 500, criterion = 'gini',
                              max_features = 'auto', max_depth = 6, random_state = 123)
rfc2.fit(X_train, y_train)
pred_rfc2 = rfc2.predict(X_test)

print(classification_report(y_test, pred_rfc2))
print(confusion_matrix(y_test, pred_rfc2))

rfc_eval2 = cross_val_score(estimator = rfc2, X = X_train, y = y_train, cv = 10)
rfc_eval2_percent = rfc_eval2.mean()*100
print(f"Random Forest Classifier Tunning the Model: {rfc_eval2_percent}")

# Tunning Support Vector Machine

param = {
    'C': [0.1,0.4,0.8,0.9,1,1.1,1.4,1.6,1.8,2],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.4,0.8,0.9,1,1.1,1.4,1.6,1.8,2]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10) # Creating the variable to grid our model
grid_svc.fit(X_train, y_train)
grid_svc.best_params_

svc2 = SVC(C = 0.9, gamma = 0.4, kernel = 'rbf', random_state = 123)

svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)

print(classification_report(y_test, pred_svc2))
print(confusion_matrix(y_test, pred_svc2))

svc_eval2 = cross_val_score(estimator = svc2, X = X_train, y = y_train, cv = 10)
svc_eval2_percent = svc_eval2.mean()*100
print(f"Support Vector Machine Tunning the Model: {svc_eval2_percent}")

##### Choosing the best model #####

models = {'dbl': dummy_baseline,
          'rfc': rfc_eval_percent, 'rfc2': rfc_eval2_percent,
          'svc': svc_eval_percent, 'svc2': svc_eval2_percent}
models
model_value = models[max(models, key=models.get)]
model_value

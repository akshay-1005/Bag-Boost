#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")
import datetime

# ## 1.1 Read the Dataset and doing Descriptive stats.

# In[2]:


transport = pd.read_csv(r"C:\Users\Akshay\Desktop\intro to phython\ML\project\Transport.csv")

transport.head()


transport.tail()


transport.rename(columns = {'Work Exp' : 'Work_Exp'}, inplace = True)



pd.DataFrame(data = [transport.mean(), transport.median(), transport.var(), transport.std()],
             index = ['Mean', 'Median', 'Variance', 'Standard_Deviation']).round(2)



pd.DataFrame(data = transport.mode())

# ## 1.2  Perform EDA


transport.shape

transport.dtypes

transport.info()

transport.isna().sum()

sns.heatmap(transport.isnull(), cbar = False, cmap = 'viridis', yticklabels = False)

missing_values = pd.DataFrame(transport.isna().sum().reset_index())
missing_values.columns = ['Features', 'Missing Count']
missing_values['% missing'] = round(missing_values['Missing Count'] / transport.shape[0] * 100, 2)
missing_values


dups = transport.duplicated()
print("The total number of duplicates present in the dataset is : ", dups.sum())


transport[['Gender', 'Engineer', 'MBA', 'Work_Exp', 'license']].nunique()

transport['Work_Exp'].unique()

print("The unique elements in Gender column is   :",transport['Gender'].unique())
print("The unique elements in Engineer column is :",transport['Engineer'].unique())
print("The unique elements in MBA column is      :",transport['MBA'].unique())
print("The unique elements in license column is  :",transport['license'].unique())

pd.DataFrame(data = [transport['Engineer'].value_counts(), transport['license'].value_counts(), transport['MBA'].value_counts()],
            index = ['Engineer', 'license', 'MBA']).T

for feature in transport.columns: 
    if transport[feature].dtype == 'object': 
        print(feature)
        print(transport[feature].value_counts())
        print('\n')


# ## Univariate Analysis

transport.describe().round(2)


## For continuous variables
cols = ['Age', 'Work_Exp', 'Salary', 'Distance', 'Engineer', 'MBA', 'license']
for col in cols:
    print(col)
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    sns.distplot(transport[col], kde = True)
    plt.axvline(transport[col].mean(), ymin = 0, ymax = 1, linewidth = 2.5, color = 'red')
    plt.axvline(transport[col].median(), ymin = 0, ymax = 1, linewidth = 2.5, color = 'green', linestyle = '--')
    plt.subplot(122)
    sns.boxplot(data = transport[col])
    
    plt.show()

# def UnivariateAnalysis_numeric(column):
#     plt.figure()
#     print("Distribution Plot of " +column)
#     print("-----------------------------------------------------------------------")
#     sns.distplot(transport[column], kde = False)
#     plt.show()
    
#     print('Box Plot of ' + column)
#     print("-----------------------------------------------------------------------")
#     ax = sns.boxplot(x = transport[column])
#     plt.show()

# df_num = transport.select_dtypes(include = ['float64', 'int64'])
# listnumericcolumns = list(df_num.columns.values)
# len(listnumericcolumns)

# for x in listnumericcolumns:
#     UnivariateAnalysis_numeric(x)


df_num_new = [c for c in transport.columns if (transport[c].dtypes != 'object') & (transport[c].nunique() > 2)]
fig, ax = plt.subplots(2, 2, figsize=(12,10))
ax = ax.flatten()

for i, col in enumerate(df_num_new):
    sns.boxplot(transport[col], ax = ax[i])
plt.suptitle('Outlier analysis Box plot excluding continuous with discrete value')
plt.tight_layout()

# Age, MBA and license are discrete variables (are categorical but marked as integer type) so they are showing such patterns in box plot.
# #### Percentage of outliers

Q1 = transport[['Age', 'Work_Exp', 'Salary', 'Distance']].quantile(0.25)
Q3 = transport[['Age', 'Work_Exp', 'Salary', 'Distance']].quantile(0.75)
IQR = Q3 - Q1
pd.DataFrame((((transport[['Age', 'Work_Exp', 'Salary', 'Distance']] 
               < (Q1 - 1.5 * IQR)) | (transport[['Age', 'Work_Exp', 'Salary', 'Distance']] 
                                     > (Q3 + 1.5 *IQR))).sum() / transport.shape[0] * 100).round(2), columns = ['% Outliers'])

pd.DataFrame(data = [transport.kurtosis(), transport.skew()], index = ['Kurtosis', 'Skew']).T.round(3)


# ### Univariate analysis for categorical columns

transport.describe(include = 'object')

plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(transport['Gender'])
plt.subplot(122)
sns.countplot(transport['Transport'])

# ### Bi-variate analysis for continuous columns.

# For continuous variables
sns.pairplot(transport, diag_kind = 'kde', hue = 'Transport')


plt.figure(figsize=(15,10))
sns.heatmap(data = transport.corr(), annot = True, mask = np.triu(transport.corr(), 1))

# ### Bi-variate analysis between target variable and independent variables


plt.figure(figsize=(15,10))

plt.subplot(221)
sns.boxplot(x = transport['Age'], y = transport['Transport'])

plt.subplot(222)
sns.boxplot(x = transport['Work_Exp'], y = transport['Transport'])

plt.subplot(223)
sns.boxplot(x = transport['Salary'], y = transport['Transport'])

plt.subplot(224)
sns.boxplot(x = transport['Distance'], y = transport['Transport'])

plt.tight_layout()
plt.show()


plt.figure(figsize=(15,13))

plt.subplot(221)
sns.boxplot(x = transport['Age'], y = transport['Transport'], hue = transport['Gender'])

plt.subplot(222)
sns.boxplot(x = transport['Work_Exp'], y = transport['Transport'], hue = transport['Gender'])

plt.subplot(223)
sns.boxplot(x = transport['Salary'], y = transport['Transport'], hue = transport['Gender'])

plt.subplot(224)
sns.boxplot(x = transport['Distance'], y = transport['Transport'], hue = transport['Gender'])

plt.tight_layout()
plt.show()


pd.DataFrame(data = transport.groupby("Transport").agg(['std', 'median'])).T


# ## 1.3

# ### Encoding the categorical Varialbe


transport.info()

trans = transport.copy()

# To convert target variable
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

## Applying the created Label Encoder object for the target class
## Assigning the 1 to Public Transport and 0 to Private transport

trans['Transport'] = LE.fit_transform(trans['Transport'])
trans.head()

trans.tail()

trans.info()

## Converting the Gender column by dummy variable
trans_dummy = pd.get_dummies(trans, drop_first = True)
trans_dummy.head()

# for feature in trans.columns: 
#     if trans_dummy[feature].dtype == 'object': 
#         print('feature:',feature, '\n')
#         print(pd.Categorical(trans_dummy[feature].unique()))
#         print(pd.Categorical(trans_dummy[feature].unique()).codes)
#         trans_dummy[feature] = pd.Categorical(trans_dummy[feature]).codes

trans_dummy.info()


trans_dummy.Transport.value_counts(normalize=True)

# #### Data Split

X = trans_dummy.drop("Transport", axis=1)

y = trans_dummy.pop("Transport")

X.head()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 1, stratify = y)


print('Dimension of X_train is :',X_train.shape)
print('Dimension of X_test is  :', X_test.shape)
print('Dimension of y_train is :',y_train.shape)
print('Dimension of y_test is  :', y_test.shape)


# #### Scaling the Data

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

# ## 1.4 Logistic Model


trans.head()


trans.info()

trans['Gender'] = LE.fit_transform(trans['Gender'])
trans.head()

import statsmodels.formula.api as SM


# ### Model 1: Logistic model with all variables

# In[54]:


model_1 = SM.logit(formula = 'Transport~Age+Gender+Engineer+MBA+Work_Exp+Salary+Distance+license', data = trans).fit()
model_1.summary()


# In[55]:


## Calculation VIF

def vif_cal(input_data):
    x_vars = input_data
    xvar_names = input_data.columns
    for i in range(0, xvar_names.shape[0]):
        y = x_vars[xvar_names[i]] 
        x = x_vars[xvar_names.drop(xvar_names[i])]
        rsq = SM.ols(formula = "y~x", data = x_vars).fit().rsquared  
        vif = round(1/(1-rsq),2)
        print(xvar_names[i], " VIF = " , vif)


# In[56]:


vif_cal(input_data = trans[['Age', 'Gender', 'Engineer', 'MBA', 'Work_Exp', 'Salary', 'Distance', 'license']])


# In[57]:


model_2 = SM.logit(formula = 'Transport~Age+Gender+Engineer+MBA+Salary+Distance+license', data = trans).fit()
model_2.summary()


# In[58]:


vif_cal(input_data = trans[['Age', 'Gender', 'Engineer', 'MBA', 'Salary', 'Distance', 'license']])

Taking threshold value for VIF as 5.
# In[59]:


## Model building after removing Engineer having p-value > 0.05
model_3 = SM.logit(formula = 'Transport~Age+Gender+MBA+Salary+Distance+license', data = trans).fit()
model_3.summary()


# In[60]:


vif_cal(input_data = trans[['Age', 'Gender', 'MBA', 'Salary', 'Distance', 'license']])


# In[61]:


## Model building after removing MBA, p-value >0.05
model_4 = SM.logit(formula = 'Transport~Age+Gender+Salary+Distance+license', data = trans).fit()
model_4.summary()


# In[62]:


vif_cal(input_data = trans[['Age','Gender', 'Salary', 'Distance', 'license']])


# In[63]:


trans.head()


# In[64]:


from sklearn.linear_model import LogisticRegression


# In[65]:


lr = LogisticRegression(solver = 'newton-cg', penalty='none')


# In[ ]:





# In[66]:


Train, Test = train_test_split(trans, test_size=0.3, random_state=1, stratify = trans['Transport'])


# In[67]:


model_t_1 = LogisticRegression(solver = 'newton-cg', penalty='none', max_iter=300)
model_t_1 = model_t_1.fit(Train[['Age', 'Gender', 'Engineer', 'MBA', 'Work_Exp', 'Salary', 'Distance', 'license']], Train['Transport'])

# Predicting on the Training Data
model_t_1_pred_train = model_t_1.predict(Train[['Age', 'Gender', 'Engineer', 'MBA', 'Work_Exp', 'Salary', 'Distance', 'license']])


# In[68]:


from sklearn.metrics import roc_auc_score, roc_curve, classification_report, accuracy_score, confusion_matrix

# cofusion matrix for training data
cnf_matrix = confusion_matrix(Train['Transport'], model_t_1_pred_train)
cnf_matrix


# In[69]:


# classification report for training data

report_1 = classification_report(Train['Transport'], model_t_1_pred_train)
print('Classification Report for Train set')
print(report_1)


# In[70]:


# Predicting on the Test Data
model_t_1_pred_test = model_t_1.predict(Test[['Age', 'Gender', 'Engineer', 'MBA', 'Work_Exp', 'Salary', 'Distance', 'license']])

# Getting probabilities for Test Data
model_t_1_pred_test_prob = model_t_1.predict_proba(Test[['Age', 'Gender', 'Engineer', 'MBA', 'Work_Exp', 'Salary', 'Distance', 'license']])[:, 1]


# In[71]:


# confusion matrix for test data
cnf_matrix_test = confusion_matrix(Test['Transport'], model_t_1_pred_test)
cnf_matrix_test


# In[72]:


report_test = classification_report(Test['Transport'], model_t_1_pred_test)
print('Classification Report for Test set')
print(report_test)


# In[73]:


plt.figure(figsize=(8,4))
plt.subplot(121)
plt.title("Model 1 Train set")
sns.heatmap(confusion_matrix(Train['Transport'], model_t_1_pred_train), annot = True, fmt = '.2f')
plt.xlabel("Predicted label")
plt.ylabel("Actual label")

plt.subplot(122)
plt.title("Model 1 Test set")
sns.heatmap(confusion_matrix(Test['Transport'], model_t_1_pred_test), annot = True)
plt.xlabel("Predicted label")
plt.ylabel("Actual label")

plt.tight_layout()
plt.show()


# In[74]:


## Final model for Descriptive i.e. model 4
model_t_4 = LogisticRegression(solver = 'newton-cg', penalty ='none', max_iter = 300)
model_t_4 = model_t_4.fit(Train[['Age', 'Gender', 'Salary', 'Distance', 'license']], Train['Transport'])

# Predicting on the Training Data
model_t_4_pred_train = model_t_4.predict(Train[['Age', 'Gender', 'Salary', 'Distance', 'license']])


# In[75]:


# Predicting on the Test Data
model_t_4_pred_test = model_t_4.predict(Test[['Age', 'Gender', 'Salary', 'Distance', 'license']])

# Getting probabilities for Test Data
model_t_4_pred_test_prob = model_t_4.predict_proba(Test[['Age', 'Gender', 'Salary', 'Distance', 'license']])[:, 1]


# In[76]:


cnf_matrix_test4 = confusion_matrix(Test['Transport'], model_t_4_pred_test)
cnf_matrix_test4


# In[77]:


report_test4 = classification_report(Train['Transport'], model_t_4_pred_train)
print('Classification Report for Train set')
print(report_test4)


# In[78]:


report_test4 = classification_report(Test['Transport'], model_t_4_pred_test)
print('Classification Report for Test set')
print(report_test4)


# In[79]:


plt.figure(figsize=(8,4))
plt.subplot(121)
plt.title("Model 4 Train set")
sns.heatmap(confusion_matrix(Train['Transport'], model_t_4_pred_train), annot = True, fmt = '.2f')
plt.xlabel("Predicted label")
plt.ylabel("Actual label")

plt.subplot(122)
plt.title("Model 4 Test set")
sns.heatmap(confusion_matrix(Test['Transport'], model_t_4_pred_test), annot = True, fmt = '.2f')
plt.xlabel("Predicted label")
plt.ylabel("Actual label")

plt.tight_layout()
plt.show()


# In[80]:


from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score


# In[81]:


plt.figure(figsize=(7,7))
plt.plot([0, 1], [0, 1], linestyle='--', c = 'r') # Reference line

fpr1, tpr1, thresholds1 = roc_curve(Test['Transport'], model_t_1_pred_test_prob) # Model 1
auc = roc_auc_score(Test['Transport'], model_t_1_pred_test_prob) # getting AUC for the model
# plot the roc curve for the model
plt.plot(fpr1, tpr1, marker = '.', label = 'mod_1: %.2f'% auc)

fpr4, tpr4, thresholds4 = roc_curve(Test['Transport'], model_t_4_pred_test_prob) # Model 2
auc = roc_auc_score(Test['Transport'], model_t_4_pred_test_prob) # getting AUC for the model
plt.plot(fpr4, tpr4, marker = '.', label = 'mod_4: %.3f'% auc)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[ ]:





# ## 1.5 KNN Model

# In[82]:


from sklearn.metrics import roc_auc_score, roc_curve, classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[83]:


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


# In[84]:


print("Using Basic Model :", '\n')
print("Train Accuracy is : {}".format(accuracy_score(y_train, knn.predict(X_train))))
print("\nTest Accuracy is : {}".format(accuracy_score(y_test, knn.predict(X_test))))
print('---------------------------------------------')

print("\nTrain ROC-AUC score is : {}".format(roc_auc_score(y_train, knn.predict_proba(X_train)[:,1])))
print("\nTest ROC-AUC score is : {}".format(roc_auc_score(y_test, knn.predict_proba(X_test)[:,1])))
print('---------------------------------------------')

print("\nConfusion matrix for  train set : ","\n",confusion_matrix(y_train, knn.predict(X_train)))
print("\nConfusion matrix for test set : ","\n",confusion_matrix(y_test, knn.predict(X_test)))


# In[85]:


from sklearn.metrics import confusion_matrix, classification_report

y_train_predict = knn.predict(X_train)
model_score = knn.score(X_train, y_train)
print("Train set Basic Model\n")
print('Classification report\n',classification_report(y_train, y_train_predict))


# In[87]:


print('\nClassification report for Test default model\n')
print(classification_report(y_test, knn.predict(X_test)))


# In[89]:


plt.figure(figsize=(15,4))
plt.subplot(121)
plt.title("Train set Basic")
sns.heatmap(confusion_matrix(y_train, y_train_predict), annot = True, fmt = '.1f')
plt.subplot(122)
plt.title("Test set Basic")
sns.heatmap(confusion_matrix(y_test, knn.predict(X_test)), annot = True, fmt = '.1f')


# In[90]:


acc_train = []
acc_test = []
for k in range(1,30, 2):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    
    y_pred_train = knn.score(X_train, y_train)
    acc_train.append(y_pred_train)
    
    y_pred_test = knn.score(X_test, y_test)
    acc_test.append(y_pred_test)
    
plt.figure(figsize=(8,5))
plt.plot(range(1,30,2) , acc_train)
plt.plot(range(1,30,2) , acc_test)
plt.xlabel('k Value')
plt.ylabel('Accuracy score')


# In[91]:


knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)


# In[ ]:





# In[93]:


from sklearn import metrics
## Performance Matrix on train data set
y_train_predict = knn.predict(X_train)
model_score = knn.score(X_train, y_train)
print('Train set for k = 9')
print('\nAccuracy :', model_score)
print('\nConfusion matrix :')
print(metrics.confusion_matrix(y_train, y_train_predict))
print('\nClassififcation report','\n')
print(metrics.classification_report(y_train, knn.predict(X_train)))


# In[94]:


## Performance Matrix on test data set
y_test_predict = knn.predict(X_test)
model_score = knn.score(X_test, y_test)
print("Test set for k = 9")
print('\nAccuracy : ', model_score)
print('\nConfusion matrix :')
print(metrics.confusion_matrix(y_test, y_test_predict))
print('\n', 'Classification report', '\n')
print(metrics.classification_report(y_test, knn.predict(X_test)))


# In[95]:


plt.figure(figsize=(15,4))
plt.subplot(121)
plt.title("Train set for k = 9")
sns.heatmap(confusion_matrix(y_train, y_train_predict), annot = True, fmt = '.1f')
plt.subplot(122)
plt.title("Test set for k = 9")
sns.heatmap(confusion_matrix(y_test, y_test_predict), annot = True, fmt = '.1f')

The accuracy for Train and Test set for 'BASE Model' for class 1 is 0.8645 and 0.8134 respectively.
The precision for the Train and Test for 'BASE Model' for class 1 is 0.85 and 0.81 respectively.
The Recall for the Train and Test set for 'BASE Model' for class 1 is 0.97 and 0.96.
The f1-score for Train and Test set for 'BASE Model' for class 1 is 0.91 and 0.86.

The accuracy for Train and Test set for 'k = 9' for class 1 is 0.81 and 0.81 respectively.
The precision for the Train and Test for 'k = 9' for class 1 is 0.80 and 0.80 respectively.
The Recall for the Train and Test set for 'k = 9' for class 1 is 0.97 and 0.96.
The f1-score for Train and Test set for 'k = 9' for class 1 is 0.87 and 0.88.

Both model is a good fit model because there is not much differecnce between the train and test set accuracies (thumb rule is there can be a maximum difference of 10% between them) 
    A overfit model is one in which the training set is performs extremly well i.e. by capturing noise of the data.
    A underfit model is one in which the test data performs better than training data and a underfit model is not suitable for     building model.
No hyperparameters is used in building the basic model.
For 2nd model n_neighbours = 9 has been taken as only hyper parameter.
# ### Hyper Tunning on KNN model

# In[96]:


params = {'n_neighbors' : [3,5,7,9,11,13], 'metric' : ['minkowski'],
          'algorithm' : ['auto', 'ball_tree','kd_tree','brute'], 'p' : [1,2],
         'leaf_size' : [5,10,15,20]}


# In[97]:


grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid = params, refit = True, verbose = 1, n_jobs = -1, cv = 3)
grid_knn.fit(X_train, y_train)


# In[98]:


grid_knn.best_params_


# In[99]:


Grid_train_predict = grid_knn.predict(X_train)
Grid_test_predict = grid_knn.predict(X_test)


# In[100]:


print("Using Hyper-Tunned Model :", '\n')
print("Train Accuracy is : {}".format(accuracy_score(y_train, Grid_train_predict)))
print("\nTest Accuracy is : {}".format(accuracy_score(y_test, Grid_test_predict)))
print('---------------------------------------------')

print("\nTrain ROC-AUC score is : {}".format(roc_auc_score(y_train, grid_knn.predict_proba(X_train)[:,1])))
print("\nTest ROC-AUC score is : {}".format(roc_auc_score(y_test, grid_knn.predict_proba(X_test)[:,1])))
print('---------------------------------------------')

print("\nConfusion matrix for  train set : ","\n",confusion_matrix(y_train, Grid_train_predict))
print("\nConfusion matrix for test set : ","\n",confusion_matrix(y_test, Grid_test_predict))


# In[101]:


print('Classification report Train set :')
print(classification_report(y_train, Grid_train_predict))


# In[102]:


print('Classification report Test set :')
print(classification_report(y_test, Grid_test_predict))


# In[103]:


plt.figure(figsize=(15,4))
plt.subplot(121)
plt.title("Train set Tunned")
sns.heatmap(confusion_matrix(y_train, Grid_train_predict), annot = True, fmt = '.1f')
plt.subplot(122)
plt.title("Test set Tunned")
sns.heatmap(confusion_matrix(y_test, Grid_test_predict), annot = True, fmt = '.1f')

The Hyper tunned model for KNN is a goodfit model and giving good accuracy score.
Hyper-parametes used - n_neighbours, metric, algorith, leaf_size and p.
# In[104]:


plt.figure(figsize=(7,7))
plt.plot([0, 1], [0, 1], linestyle='--', c = 'r') # Reference line

fpr1, tpr1, thresholds1 = roc_curve(y_test, knn.predict_proba(X_test)[:,1]) # Model 1
auc = roc_auc_score(y_test, knn.predict_proba(X_test)[:,1]) # getting AUC for the model
# plot the roc curve for the model
plt.plot(fpr1, tpr1, marker = '.', label = 'Base : %.2f'% auc)

fpr2, tpr2, thresholds2 = roc_curve(y_test, knn.predict_proba(X_test)[:,1]) # Model 2
auc = roc_auc_score(y_test, knn.predict_proba(X_test)[:,1]) # getting AUC for the model
plt.plot(fpr2, tpr2, marker = '.', label = 'k=9 : %.3f'% auc)

fpr3, tpr3, thresholds3 = roc_curve(y_test, grid_knn.predict_proba(X_test)[:,1]) # Model 2
auc = roc_auc_score(y_test, grid_knn.predict_proba(X_test)[:,1]) # getting AUC for the model
plt.plot(fpr3, tpr3, marker = '.', label = 'Hyper-Tunned: %.3f'% auc)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# ## 1.6 Bagging and Boosting

# ### Bagging

# In[105]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[106]:


## Bagging with default values
bag= BaggingClassifier(random_state= 1)
bag.fit(X_train, y_train)


# In[107]:


print("Bagging Basic Model")
print("\nTrain Accuracy is : {}".format(accuracy_score(y_train, bag.predict(X_train))))
print("Test Accuracy is : {}".format(accuracy_score(y_test, bag.predict(X_test))))
print('------------------------------------------------')

print("\nTrain ROC-AUC score is : {}".format(roc_auc_score(y_train, bag.predict_proba(X_train)[:,1])))
print("Test ROC-AUC score is : {}".format(roc_auc_score(y_test, bag.predict_proba(X_test)[:,1])))
print('------------------------------------------------')

print("\nConfusion matrix for train set : ","\n",confusion_matrix(y_train, bag.predict(X_train)))
print("Confusion matrix for test set : ","\n",confusion_matrix(y_test, bag.predict(X_test)))


# In[108]:


print('Classification report Train set :')
print(classification_report(y_train, bag.predict(X_train)))


# In[109]:


print('Classification report Test set :')
print(classification_report(y_test, bag.predict(X_test)))


# In[110]:


plt.figure(figsize=(15,4))
plt.subplot(121)
plt.title("Train set Basic")
sns.heatmap(confusion_matrix(y_train, bag.predict(X_train)), annot = True, fmt = '.1f')
plt.subplot(122)
plt.title("Test set Basic")
sns.heatmap(confusion_matrix(y_test, bag.predict(X_test)), annot = True, fmt = '.1f')

The model is overfit as the training set is performing extremly good i.e. training accuracy is 100%
not a good fit model as there is a difference of 18% between train and test set using default parameters.
# ### Hyperparameter tuning of Bagging Classifier -

# In[111]:


param_bag = {'base_estimator':[LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()],
             'max_samples':[0.3,0.5,0.7,0.9], 'n_estimators' :[10,30,50,100,501],
             'max_features': [1,2,3,4,6,7,8,9], 'random_state': [1]}


# In[112]:


grid_bag = GridSearchCV(BaggingClassifier(), param_grid = param_bag, refit = True, verbose = True, n_jobs = -1, cv = 3)
grid_bag.fit(X_train, y_train)


# In[113]:


grid_bag.best_params_


# In[114]:


print('Bagging Tunned with base estimator: LR, DT and RF')
print("\nTrain Accuracy is : {}".format(accuracy_score(y_train, grid_bag.predict(X_train))))
print("\nTest Accuracy is : {}".format(accuracy_score(y_test, grid_bag.predict(X_test))))
print('------------------------------------------------')

print("\nTrain ROC-AUC score is : {}".format(roc_auc_score(y_train, grid_bag.predict_proba(X_train)[:,1])))
print("Test ROC-AUC score is : {}".format(roc_auc_score(y_test, grid_bag.predict_proba(X_test)[:,1])))
print('------------------------------------------------')

print("\nConfusion matrix for train set : ","\n",confusion_matrix(y_train, grid_bag.predict(X_train)))
print("Confusion matrix for test set : ","\n",confusion_matrix(y_test, grid_bag.predict(X_test)))


# In[115]:


print('Classification report Train set :')
print(classification_report(y_train,grid_bag.predict(X_train)))


# In[116]:


print('Classification report Test set :')
print(classification_report(y_test,grid_bag.predict(X_test)))


# In[117]:


auc_basic = roc_auc_score(y_test, bag.predict_proba(X_test)[:,1])
auc_tuned = roc_auc_score(y_test, grid_bag.predict_proba(X_test)[:,1])


# In[118]:


# plt.figure(figsize=(7,7))
# plt.plot([0, 1], [0, 1], linestyle='--', c = 'r') # Reference line
# plt.title("Bagging Classifier Basic and Tunned comparision")
# # calculate roc curve for base model
# lr_fpr, lr_tpr, lr_threshold = roc_curve(y_test, bag.predict_proba(X_test)[:,1])

# # plot the roc curve for the model
# plt.plot(lr_fpr, lr_tpr, marker='.', label='Base : %.3f'% auc_basic)

# # roc curve for tunned model
# lr_fpr, lr_tpr, lr_threshold = roc_curve(y_test, grid_bag.predict_proba(X_test)[:,1])
# # plot the roc curve for the model
# plt.plot(lr_fpr, lr_tpr, marker='.', label='Tunned : %.3f'% auc_tuned)

# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# plt.show()


# ### Bagging with Random Forest as only Base estimator

# In[119]:


param_rf = {'base_estimator':[RandomForestClassifier()],'max_samples':[0.6,0.7],'max_features':[0.2,0.5,0.6],
             'n_estimators' :[50,100,151],'random_state': [1]}


# In[120]:


grid_bag_rf = GridSearchCV(BaggingClassifier(), param_grid = param_rf, refit = True, verbose = True, n_jobs = -1, cv = 3)
grid_bag_rf.fit(X_train, y_train)


# In[121]:


grid_bag_rf.best_params_


# In[122]:


print("Bagging [Random Forest]")
print("\nTrain Accuracy is : {}".format(accuracy_score(y_train,grid_bag_rf.predict(X_train))))
print("Test Accuracy is    : {}".format(accuracy_score(y_test,grid_bag_rf.predict(X_test))))
print('------------------------------------------------')

print("\nTrain ROC-AUC score is : {}".format(roc_auc_score(y_train,grid_bag_rf.predict_proba(X_train)[:,1])))
print("Test ROC-AUC score is    : {}".format(roc_auc_score(y_test,grid_bag_rf.predict_proba(X_test)[:,1])))
print('-------------------------------------------------')

print("\nConfusion matrix for train set : ","\n",confusion_matrix(y_train,grid_bag_rf.predict(X_train)))
print("Confusion matrix for test set    : ","\n",confusion_matrix(y_test,grid_bag_rf.predict(X_test)))


# In[123]:


print('Classification report Train set :')
print(classification_report(y_train,grid_bag_rf.predict(X_train)))


# In[124]:


print('Classification report Test set :')
print(classification_report(y_test,grid_bag_rf.predict(X_test)))


# In[125]:


plt.figure(figsize=(15,4))
plt.subplot(121)
plt.title("Train set Bagging (RF)")
sns.heatmap(confusion_matrix(y_train, grid_bag_rf.predict(X_train)), annot = True, fmt = '.1f')
plt.subplot(122)
plt.title("Test set Bagging (RF)")
sns.heatmap(confusion_matrix(y_test, grid_bag_rf.predict(X_test)), annot = True, fmt = '.1f')


# In[126]:


auc_basic = roc_auc_score(y_test, bag.predict_proba(X_test)[:,1])
auc_tuned = roc_auc_score(y_test, grid_bag.predict_proba(X_test)[:,1])
auc_tunedRF = roc_auc_score(y_test, grid_bag_rf.predict_proba(X_test)[:,1])


# In[127]:


plt.figure(figsize=(7,7))
plt.plot([0, 1], [0, 1], linestyle='--', c = 'r') # Reference line
plt.title("Bagging Model comparision")

# calculate roc curve for base model
lr_fpr, lr_tpr, lr_threshold = roc_curve(y_test, bag.predict_proba(X_test)[:,1])
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='Base : %.3f'% auc_basic)

# roc curve for tunned model
lr_fpr1, lr_tpr1, lr_threshold1 = roc_curve(y_test, grid_bag.predict_proba(X_test)[:,1])
# plot the roc curve for the model
plt.plot(lr_fpr1, lr_tpr1, marker='.', label='Tunned DT : %.3f'% auc_tuned)

# roc curve for hyper-tunned model
lr_fpr, lr_tpr, lr_threshold = roc_curve(y_test, grid_bag_rf.predict_proba(X_test)[:,1])
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='Tunned RF : %.3f'% auc_tunedRF)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# ## Random Forest

# In[128]:



RF=RandomForestClassifier()

#Train the model using the training sets y_pred = clf.predict(X_test)
RF.fit(X_train, y_train)


# In[129]:


y_pred = RF.predict(X_test)

model_trainRF = RF.score(X_train, y_train)
model_testRF = RF.score(X_test, y_test)


# In[130]:


print("Random Forest BASE model")
print("-------------------------------------------------")
print("\nTrain accuracy :", model_trainRF)
print("Test accuracy  :", model_testRF)
print('-------------------------------------------------')
print("\nTrain ROC-AUC score is : {}".format(roc_auc_score(y_train, RF.predict_proba(X_train)[:,1])))
print("Test ROC-AUC score is    : {}".format(roc_auc_score(y_test, RF.predict_proba(X_test)[:,1])))
print('-------------------------------------------------')
print('\nConfusion matrix train model\n', metrics.confusion_matrix(y_train, RF.predict(X_train)))
print('\nConfusion matrix test model\n', metrics.confusion_matrix(y_test, y_pred))


# In[131]:


plt.figure(figsize=(15,4))
plt.subplot(121)
plt.title("Train set Basic RF")
sns.heatmap(metrics.confusion_matrix(y_train, RF.predict(X_train)), annot = True, fmt = '.1f')
plt.subplot(122)
plt.title("Test set Basic RF")
sns.heatmap(metrics.confusion_matrix(y_test, RF.predict(X_test)), annot = True, fmt = '.1f')


# In[132]:


param = {'max_depth': [7, 10],
         'max_features': [4, 6],
         'min_samples_leaf': [50, 100],
         'min_samples_split': [150, 300],
         'n_estimators': [50, 100,301]}


# In[133]:


rfcl = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rfcl, param_grid = param, cv = 3, verbose = 1)

grid_search.fit(X_train, y_train)


# In[134]:


grid_search.best_params_


# In[135]:


best_grid = grid_search.best_estimator_


# In[136]:


ytrain_predict = best_grid.predict(X_train)
ytest_predict = best_grid.predict(X_test)


# In[137]:


print("Random Forest tunned model")
print("--------------------------------------------------")
print("\nTrain accuracy :", best_grid.score(X_train, y_train))
print("Test accuracy  :", best_grid.score(X_test, y_test))
print('----------------------------------------------------')
print("\nTrain ROC-AUC score is : {}".format(roc_auc_score(y_train, best_grid.predict_proba(X_train)[:,1])))
print("Test ROC-AUC score is    : {}".format(roc_auc_score(y_test, best_grid.predict_proba(X_test)[:,1])))
print('---------------------------------------------------')
print('\nConfusion matrix train model\n', metrics.confusion_matrix(y_train, best_grid.predict(X_train)))
print('\nConfusion matrix test model\n', metrics.confusion_matrix(y_test, best_grid.predict(X_test)))


# In[138]:


print('Classification report Train set :\n')
print(classification_report(y_train, best_grid.predict(X_train)))


# In[139]:


print('Classification report Test set :\n')
print(classification_report(y_test, best_grid.predict(X_test)))


# In[140]:


plt.figure(figsize=(15,4))
plt.subplot(121)
plt.title("Train set Tunned RF")
sns.heatmap(metrics.confusion_matrix(y_train, best_grid.predict(X_train)), annot = True, fmt = '.1f')
plt.subplot(122)
plt.title("Test set Tunned RF")
sns.heatmap(metrics.confusion_matrix(y_test, best_grid.predict(X_test)), annot = True, fmt = '.1f')


# In[141]:


Importance = pd.DataFrame(RF.feature_importances_*100, index = X.columns).sort_values(by = 0, ascending = False)
plt.figure(figsize=(12,7))
sns.barplot(Importance[0], Importance.index, palette = 'rainbow')
plt.ylabel('Feature Name')
plt.xlabel('Feature Importance in %')
plt.title('Feature Importance Plot')
plt.show()


# In[142]:


auc_basic = roc_auc_score(y_test, RF.predict_proba(X_test)[:,1])
auc_tuned = roc_auc_score(y_test, best_grid.predict_proba(X_test)[:,1])


# In[143]:


plt.figure(figsize=(7,7))
plt.plot([0, 1], [0, 1], linestyle='--', c = 'r') # Reference line
plt.title("Random Forest Model comparision")

# calculate roc curve for base model
lr_fpr, lr_tpr, lr_threshold = roc_curve(y_test, RF.predict_proba(X_test)[:,1])
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='Base : %.3f'% auc_basic)

# roc curve for tunned model
lr_fpr1, lr_tpr1, lr_threshold1 = roc_curve(y_test, best_grid.predict_proba(X_test)[:,1])
# plot the roc curve for the model
plt.plot(lr_fpr1, lr_tpr1, marker='.', label='Tunned RF : %.3f'% auc_tuned)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[ ]:





# ### Boosting
# ### Gradient Boosting

# In[144]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(random_state = 1)
gbc.fit(X_train, y_train)


# In[145]:


print("Gradient Boosting Basic")
print("\nTrain Accuracy is : {}".format(accuracy_score(y_train, gbc.predict(X_train))))
print("Test Accuracy is  : {}".format(accuracy_score(y_test, gbc.predict(X_test))))
print('-------------------------------------------------')

print("\nTrain ROC-AUC score is : {}".format(roc_auc_score(y_train, gbc.predict_proba(X_train)[:,1])))
print("Test ROC-AUC score is    : {}".format(roc_auc_score(y_test, gbc.predict_proba(X_test)[:,1])))
print('--------------------------------------------------')

print("\nConfusion matrix for train set : ","\n",confusion_matrix(y_train, gbc.predict(X_train)))
print("\nConfusion matrix for test set  : ","\n",confusion_matrix(y_test, gbc.predict(X_test)))


# In[146]:


print('Classification report Train set :')
print(classification_report(y_train, gbc.predict(X_train)))


# In[147]:


print('Classification report Test set :')
print(classification_report(y_test, gbc.predict(X_test)))


# In[148]:


plt.figure(figsize=(15,4))
plt.subplot(121)
plt.title("Train set Gradient Boosting Basic")
sns.heatmap(metrics.confusion_matrix(y_train, gbc.predict(X_train)), annot = True, fmt = '.1f')
plt.subplot(122)
plt.title("Test set Gradient Boosting Basic")
sns.heatmap(metrics.confusion_matrix(y_test, gbc.predict(X_test)), annot = True, fmt = '.1f')


# #### Hyperparameter tuning of Gradient Boosting Classifier-

# In[149]:


param_gbc={'learning_rate': [0.01,0.02,0.05], 'max_depth':[3,5,7],
            'min_samples_split': [7,9,11,13], 'n_estimators':[100,200],'random_state': [1]}    


# In[150]:


grid_gbc_2 = GridSearchCV(GradientBoostingClassifier(), param_grid = param_gbc, refit = True, verbose = True,
                          n_jobs = -1, cv = 3)
grid_gbc_2.fit(X_train, y_train)


# In[151]:


grid_gbc_2.best_params_


# In[152]:


print("Gradient Boosting Tunned")
print("\nTrain Accuracy is : {}".format(accuracy_score(y_train, grid_gbc_2.predict(X_train))))
print("\nTest Accuracy is : {}".format(accuracy_score(y_test, grid_gbc_2.predict(X_test))))
print('------------------------------------------------')

print("\nTrain ROC-AUC score is : {}".format(roc_auc_score(y_train, grid_gbc_2.predict_proba(X_train)[:,1])))
print("\nTest ROC-AUC score is : {}".format(roc_auc_score(y_test, grid_gbc_2.predict_proba(X_test)[:,1])))
print('-------------------------------------------------')

print("\nConfusion matrix for train set : ","\n",confusion_matrix(y_train, grid_gbc_2.predict(X_train)))
print("\nConfusion matrix for test set : ","\n",confusion_matrix(y_test, grid_gbc_2.predict(X_test)))


# In[153]:


print('Classification report Train set :', '\n')
print(classification_report(y_train, grid_gbc_2.predict(X_train)))


# In[154]:


print('Classification report Test set :', '\n')
print(classification_report(y_test,grid_gbc_2.predict(X_test)))


# In[155]:


plt.figure(figsize=(15,4))
plt.subplot(121)
plt.title("Train set Tunned Gradinet Boosting")
sns.heatmap(metrics.confusion_matrix(y_train, grid_gbc_2.predict(X_train)), annot = True, fmt = '.1f')
plt.subplot(122)
plt.title("Test set Tunned Gradient Boosting")
sns.heatmap(metrics.confusion_matrix(y_test, grid_gbc_2.predict(X_test)), annot = True, fmt = '.1f')


# In[156]:


auc_basic = roc_auc_score(y_test, gbc.predict_proba(X_test)[:,1])
auc_tun = roc_auc_score(y_test, grid_gbc_2.predict_proba(X_test)[:,1])


# In[157]:


plt.figure(figsize=(7,7))
plt.plot([0, 1], [0, 1], linestyle='--', c = 'r') # Reference line

# calculate roc curve for base model
lr_fpr, lr_tpr, lr_threshold = roc_curve(y_test, gbc.predict_proba(X_test)[:,1])

# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='Base : %.3f'% auc_basic)

# roc curve for tunned model
lr_fpr, lr_tpr, lr_threshold = roc_curve(y_test, grid_gbc_2.predict_proba(X_test)[:,1])
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='Tunned : %.3f'% auc_tun)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[ ]:





# ### Adaptive boosting: ADABoost

# In[158]:


from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)


# In[159]:


print("Train Accuracy is : {}".format(accuracy_score(y_train, ada.predict(X_train))))
print("\nTest Accuracy is : {}".format(accuracy_score(y_test, ada.predict(X_test))))
print('------------------------------------------------')

print("\nTrain ROC-AUC score is : {}".format(roc_auc_score(y_train, ada.predict_proba(X_train)[:,1])))
print("\nTest ROC-AUC score is : {}".format(roc_auc_score(y_test, ada.predict_proba(X_test)[:,1])))
print('-------------------------------------------------')

print("\nConfusion matrix for train set : ","\n",confusion_matrix(y_train, ada.predict(X_train)))
print("\nConfusion matrix for test set : ","\n",confusion_matrix(y_test, ada.predict(X_test)))


# In[160]:


print('Classification report Train set basic:', '\n')
print(classification_report(y_train, ada.predict(X_train)))


# In[161]:


print('Classification report Test set basic :', '\n')
print(classification_report(y_test, ada.predict(X_test)))


# In[162]:


plt.figure(figsize=(15,4))
plt.subplot(121)
plt.title("Train set AdaBoost Basic")
sns.heatmap(metrics.confusion_matrix(y_train, ada.predict(X_train)), annot = True, fmt = '.1f')
plt.subplot(122)
plt.title("Test set AdaBoost Basic")
sns.heatmap(metrics.confusion_matrix(y_test, ada.predict(X_test)), annot = True, fmt = '.1f')


# #### HyperParameter tunning :

# In[163]:


param_ada = {'learning_rate': [1,2,3,6,8],
            'n_estimators' : [50,100], 'random_state': [1]}    


# In[164]:


grid_ada = GridSearchCV(AdaBoostClassifier(), param_grid = param_ada, refit = True, verbose = True, n_jobs = -1, cv = 3)
grid_ada.fit(X_train, y_train)


# In[165]:


grid_ada.best_params_


# In[166]:


print("Train Accuracy is : {}".format(accuracy_score(y_train, grid_ada.predict(X_train))))
print("\nTest Accuracy is : {}".format(accuracy_score(y_test, grid_ada.predict(X_test))))
print('---------------------------------------------')

print("\nTrain ROC-AUC score is : {}".format(roc_auc_score(y_train, grid_ada.predict_proba(X_train)[:,1])))
print("\nTest ROC-AUC score is : {}".format(roc_auc_score(y_test, grid_ada.predict_proba(X_test)[:,1])))
print('-------------------------------------------------')

print("\nConfusion matrix for train set : ","\n",confusion_matrix(y_train, grid_ada.predict(X_train)))
print("\nConfusion matrix for test set : ","\n",confusion_matrix(y_test, grid_ada.predict(X_test)))


# In[167]:


print('Classification report Train set :', '\n')
print(classification_report(y_train, grid_ada.predict(X_train)))


# In[168]:


print('Classification report Test set :', '\n')
print(classification_report(y_test, grid_ada.predict(X_test)))


# In[169]:


plt.figure(figsize=(15,4))
plt.subplot(121)
plt.title("Train set AdaBoost Tunned")
sns.heatmap(metrics.confusion_matrix(y_train, grid_ada.predict(X_train)), annot = True, fmt = '.1f')
plt.subplot(122)
plt.title("Test set AdaBoost Tunned")
sns.heatmap(metrics.confusion_matrix(y_test, grid_ada.predict(X_test)), annot = True, fmt = '.1f')


# In[170]:


auc_basic = roc_auc_score(y_test, ada.predict_proba(X_test)[:,1])
auc_tun = roc_auc_score(y_test, grid_ada.predict_proba(X_test)[:,1])


# In[171]:


plt.figure(figsize=(7,7))
plt.plot([0, 1], [0, 1], linestyle='--', c = 'r') # Reference line

#Basic
lr_probs = ada.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('Boosting Classifier Basic: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='Base : %.3f'% auc_basic)

#tunned
lr_probs = grid_ada.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('Boosting classifier tunned: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='Tunned : %.3f'% auc_tun)

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


# ## Model Comparision

# In[172]:


plt.figure(figsize=(7,7))
plt.plot([0, 1], [0, 1], linestyle='--', c = 'r') # Reference line
plt.title("Model Comparision")

# Logistic
fpr4, tpr4, thresholds4 = roc_curve(Test['Transport'], model_t_4_pred_test_prob)
auc = roc_auc_score(Test['Transport'], model_t_4_pred_test_prob)
plt.plot(fpr4, tpr4, marker = '.', label = 'Logistic: %.3f'% auc)

#KNN
fpr3, tpr3, thresholds3 = roc_curve(y_test, grid_knn.predict_proba(X_test)[:,1])
auc = roc_auc_score(y_test, grid_knn.predict_proba(X_test)[:,1])
plt.plot(fpr3, tpr3, marker = '.', label = 'KNN: %.3f'% auc)

#Bagging
lr_fpr, lr_tpr, lr_threshold = roc_curve(y_test, grid_bag_rf.predict_proba(X_test)[:,1])
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='Bagging RF : %.3f'% auc_tunedRF)

#RF
lr_fpr1, lr_tpr1, lr_threshold1 = roc_curve(y_test, best_grid.predict_proba(X_test)[:,1])
# plot the roc curve for the model
plt.plot(lr_fpr1, lr_tpr1, marker='.', label='RF : %.3f'% auc_tuned)

#Gradient Boost
lr_fpr, lr_tpr, lr_threshold = roc_curve(y_test, grid_gbc_2.predict_proba(X_test)[:,1])
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='Gradient : %.3f'% auc_tun)

# AdaBoost
lr_probs = grid_ada.predict_proba(X_test)
lr_probs = lr_probs[:, 1]
# calculate scores
lr_auc = roc_auc_score(y_test, lr_probs)
# calculate roc curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='AdaBoost : %.3f'% lr_auc)


plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[173]:


# from sklearn import model_selection
# from sklearn.model_selection import cross_val_score


# In[174]:


# models = []
# models.append(('LR' , LogisticRegression(max_iter = 10000)))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('Bagging', BaggingClassifier()))
# models.append(('Random forest', RandomForestClassifier()))
# models.append(('AdaBoost', AdaBoostClassifier()))
# models.append(('Gradient Boost', GradientBoostingClassifier()))


# In[175]:


# results = []
# names = []

# for name, model in models:
#   kfold = model_selection.KFold(n_splits = 7)
#   cv_results = model_selection.cross_val_score(model, X_train , y_train , cv = kfold , scoring = 'accuracy' )
#   results.append(cv_results)
#   names.append(name)
#   msg = "%s: %f (%f)" % (name, cv_results.mean() , cv_results.std())
#   print(msg)


# In[178]:


# results = []
# names = []

# for name, model in models:
#   kfold = model_selection.KFold(n_splits = 7)
#   cv_results = model_selection.cross_val_score(model, X_test , y_test , cv = kfold , scoring = 'accuracy' )
#   results.append(cv_results)
#   names.append(name)
#   msg = "%s: %f (%f)" % (name, cv_results.mean() , cv_results.std())
# print(msg)






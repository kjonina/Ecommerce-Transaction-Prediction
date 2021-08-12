""" 
Karina Jonina 10543032
Pauline Lloyd 10525563
Stephen Quinn 10537441

B8IT106 - Tools for Data Analytics (DBS)

Assignment 2 - 20% Group Assignment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE  # imblearn library can be installed using pip install imblearn
from sklearn.ensemble import RandomForestClassifier

## Importing dataset and examining it
dataset = pd.read_csv("e-shop.csv")
pd.set_option('display.max_columns', None) # Will ensure that all columns are displayed
print(dataset.head())


print(dataset.shape) # returns no: rows and columns

print(dataset.info()) # shows data type by attribute/data types - including 2 objects that we need to convert

print(dataset.describe()) 

print(dataset.ExitRate.describe()) # shows stats of one particular column


# =============================================================================
# Converting important features
# =============================================================================

##Converting Categorical features into Numerical features 

def converter(column):
    if column == 'New_Visitor':
        return 1
    else:
        return 0

dataset['VisitorType'] = dataset['VisitorType'].apply(converter)
print(dataset.info()) # shows visitor types converted to int
print(dataset.head(2)) # prints first 2 rows
print(dataset.dtypes) # check that those 3 variables are converted

## Then convert the Month 
## get_dummies method which will split month out into
## a column for each month and assign a 1 (occured that month) or a zero (did not occur)
## will turn each month into a new columnm marked 1 if it applies
categorical_features = ['Month']
final_data = pd.get_dummies(dataset, columns = categorical_features)
print(final_data.info())
print(final_data.head(2))

##check that all data types have been converted to ints or floats
print(final_data.dtypes) # will see a col for each month now

## Dividing dataset into label and feature sets
X = final_data.drop('Transaction', axis = 1) #axis = 1 means we are dropping a column - as this is the 
## target variable, 1 denotes column, 0 denotes row (Just Indepedent Variables - so drop the target var)
Y = final_data['Transaction'] # Labels
print(type(X))
print(type(Y))
print(X.shape) # so we now rows shown and have 22 independent variables (14 + additional var when month split - the target var = 22)
print(Y.shape) # total rows and a single column


## Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X) #this is now a normalised feature set

## Dividing dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

print(X_train.shape) #70 % chosen random set
print(X_test.shape) # 30 % randomly set test set

## Implementing Oversampling to balance the dataset; SMOTE stands for Synthetic Minority Oversampling TEchnique
print("Number of observations in each class before oversampling (training data): \n", pd.Series(Y_train).value_counts())
## needed because there's a large imbAlance in the class Transaction (TRUE) and (FALSE)
## WHICH WOULD RESULT IN IMBALANCED LEARNING - so it will not learn much about the minority class - so it will become good at 
## predicting majority class but not minority class
## above shows that if we have 8571 training data set, this is currently split by FALSE 7248 and TRUE 
## 1323, THIS IS CLEARLY IMBALANCED,SO PROCEED TO BALANCE:
##  Balance the training set only - not the test set - because in the real world the data set will be imbalanced
## Usually we upsize the minority class - rarely downsize - creates artificial samples that are similar to existing ones
# and inserts them into the exisiting samples
smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)

print("Number of observations in each class after oversampling (training data): \n", pd.Series(Y_train).value_counts())
## ABOVE RETURNS BALANCE DATA SET 7248 EACH

# =============================================================================
# BUILDING A CLASSIFICATION DECISION TREE MODEL
# =============================================================================
## Building Classification Decision Tree Model
dtree = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5) 
## ENTROPY MEANS USING INFORMATION GAME TO SPLIT VARS .  Max depth only going to 5 levels, otherwise get overfit
## we don't know what is the optimal level - so experiment, we start with 5
dtree.fit(X_train, Y_train)
featimp = pd.Series(dtree.feature_importances_, index=list(X)).sort_values(ascending=False)
## tells us which features are most important in predicting which web visits result in a transaction
## var with values 0 means they weren't use to construct the decision trees, because we went to level 5
print(featimp)

## Evaluating Decision Tree Model by constructing a decision tree matrix  
Y_pred = dtree.predict(X_test)
print("Prediction Accuracy: ", metrics.accuracy_score(Y_test, Y_pred)) 
conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])
## returns: Features importance
##PageValue                  0.758752
##Month_Nov                  0.087211
##Administrative_Duration    0.030391
##Administrative             0.026385
##BounceRate                 0.020600
##Month_May                  0.020420
##VisitorType                0.017694
##Month_Mar                  0.014593
##ExitRate                   0.012125
##ProductRelated             0.005917
##ProductRelated_Duration    0.004613
##Informational              0.000809
##Month_Aug                  0.000491
##Month_Feb                  0.000000
##Month_Dec                  0.000000
##Weekend                    0.000000
##Month_Oct                  0.000000
##SpecialDay                 0.000000
##Month_Jul                  0.000000
##Month_June                 0.000000
##Informational_Duration     0.000000
##Month_Sep                  0.000000
##Prediction Accuracy:  0.84676102340773
## the prediction accuracy is totally misleading because our test set is imbalanced.  
## Do not use prediction accuracy with an imbalanced TEST 
## set, so we minimise errors  (if you min FP you'll increase FN and vice versa)
## TP:  462 accurately predicted TRUE (resulted in transaction)
## TN:  2649
## FP:  456 predicted TRUE but actually FALSE (did not result in transaction) - almost 14% of set
## classified as predicting a transaction when it would not
## FN:  107
#  
## Our objective is min FP because we want to correctly identify which visits are likely to result in a transaction
## so we want to capture the reasons for transaction success and build our website/guide traffic - with those 
## features maximised.  There are almost as many FP as there are TP, so we need to min FP

## Tuning the tree size parameter 'max_depth' and implementing cross-validation using Grid Search
## we need to find the optimal depth to minimise our FP
classifier = tree.DecisionTreeClassifier(criterion = 'entropy')
grid_param = {'max_depth': [2,3,4,5,10,15,20,25,30,35]}

gd_sr = GridSearchCV(estimator=classifier, param_grid=grid_param, scoring='precision', cv=5)
## cv = cross validation, evaluate model on a number of test sets so ensure it consistently behaves the 
## same way.  It devides the dataset it into 5 equal folds parts), 1 will become the test set, 4 parts training set.
## then it rolls again making sure that each fold becomes a test set while the others are training sets.
## if consistent, implies its a good model.  It will do this for each max depth of decision trees listed above
## it will then pick the max no: of decision trees which gives the best accurate model result
##In the above GridSearchCV(), scoring parameter should be set as follows:
##scoring = 'accuracy' when you want to maximize prediction accuracy - not appropriate for imbalanced test sets - in this case
##scoring = 'recall' when you want to minimize false negatives
##scoring = 'precision' when you want to minimize false positives
##scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
##  IF you can decide which is worse FN/FP then select balance F1
#
gd_sr.fit(X_train, Y_train)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)
## results: returns max depth and prediction accuracy
## run again results returns a different set of results
## run again results and another set of results

## Why are they different?  Because it selects a var at random and constructs decision level,
## where ties among nodes of same entropy or gain score
## this choice will create an entirely different tree - this issue occurs at any node
## this is an unreliable algorithm.  It only thinks of the node to split on , not the entire tree
## so you'll never find an optimal decision tree - so build many trees and take majority vote of the results

## Ensemble Method: Building random forest model with 300 decision trees (having a guess at how many decision trees to use)
## you need multiple training sets for this - construct bootstrap samples - sample with replacement - multiple times to create
##several training sets of value N - they aren't cut/they grow fully.  


# =============================================================================
# Random Forest - First Model
# =============================================================================
## we have 22 vars, each decision tree will be split at each node with sq rt 22 = 4.69 features at each node
##rfc = RandomForestClassifier(n_estimators=300, criterion='entropy', max_features='auto')
##can use auto - which is sq root of predictors, or 'default'
##rfc.fit(X_train,Y_train)
##Y_pred = rfc.predict(X_test)
##conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
##plt.figure(figsize=(8,6))
##sns.heatmap(conf_mat,annot=True)
##plt.title("Confusion_matrix")
##plt.xlabel("Predicted Class")
##plt.ylabel("Actual class")
##plt.show()
##print('Confusion matrix: \n', conf_mat)
##print('TP: ', conf_mat[1,1])
##print('TN: ', conf_mat[0,0])
##print('FP: ', conf_mat[0,1])
##print('FN: ', conf_mat[1,0])

##Confusion matrix results are different each time this is run as they are 
##randomly generated trees each time
##we don't know if this is the optimal no: of decision treees


# =============================================================================
# Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
# =============================================================================

rfc = RandomForestClassifier(criterion='entropy', max_features='auto', random_state=1)
grid_param = {'n_estimators': [200, 250, 300, 350, 400, 450]}

gd_sr = GridSearchCV(estimator=rfc, param_grid=grid_param, scoring='precision', cv=5)

##"""
##In the above GridSearchCV(), scoring parameter should be set as follows:
##scoring = 'accuracy' when you want to maximize prediction accuracy
##scoring = 'recall' when you want to minimize false negatives
##scoring = 'precision' when you want to minimize false positives
##scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
##"""
#
gd_sr.fit(X_train, Y_train)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)
###

##Returns the optimal no: of decision trees as 350.  
## Building random forest using the tuned parameter of 350
rfc = RandomForestClassifier(n_estimators=350, criterion='entropy', max_features='auto', random_state=1)
rfc.fit(X_train,Y_train)
featimp = pd.Series(rfc.feature_importances_, index=list(X)).sort_values(ascending=False)
print(featimp)
#
Y_pred = rfc.predict(X_test)
conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])
print("Prediction Accuracy: ", metrics.accuracy_score(Y_test, Y_pred)) 
##returns Confusion matrix: 
##TP:  423
##TN:  2805
##FP:  300
##FN:  146
###FEATURES IMPORTANCE LIST - THIS LIST CHANGES ACCORDING TO HOW MANY TREES YOU GROW.  
##RESULTS BELOW FOR OUR OPTIMAL NO: 350
##PageValue                  0.376580
##ExitRate                   0.088914
##Administrative             0.083279
##ProductRelated_Duration    0.082163
##ProductRelated             0.077825
##Administrative_Duration    0.065467
##BounceRate                 0.060118
##Month_Nov                  0.034166
##Informational              0.027919
##Informational_Duration     0.021513
##Month_May                  0.014578
##VisitorType                0.014257
##Month_Mar                  0.010772
##Weekend                    0.010717
##Month_Dec                  0.006474
##SpecialDay                 0.006174
##Month_Sep                  0.004998
##Month_Oct                  0.004145
##Month_Jul                  0.004060
##Month_Aug                  0.003525
##Month_June                 0.001716
##Month_Feb                  0.000640

##note that all var are used to construct the 350 decision trees
### Selecting features with higher significance and redefining feature set

# =============================================================================
# TRY TOP 5
# =============================================================================
X = final_data[['PageValue', 'ExitRate', 'Administrative', 'ProductRelated_Duration','ProductRelated']]
## x is redfined for min feature
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

## Dividing dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)
##show the split
print(X_train.shape) #70 % chosen random set - print statements inserted to check the split.
print(X_test.shape) # 30 % 
##returns X_train  8,571, 5 
##returns X_test  3,674, 5

##oversample
smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)
##returns (14496, 5) 
##returns (3674, 5)
##this amounts to 18,170 samples as we increased our sample size by 7,248-1,323 through
##oversampling

rfc = RandomForestClassifier(n_estimators=350, criterion='entropy', max_features='auto', random_state=1)
rfc.fit(X_train,Y_train)

Y_pred = rfc.predict(X_test)
conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Top 5')
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])
##returns
##TP:  409
##TN:  2789
##FP:  316
##FN:  160


# =============================================================================
# TRY TOP 6 - explanations of steps as above in Top 5
# =============================================================================
X = final_data[['PageValue', 'ExitRate', 'Administrative','ProductRelated_Duration', 'ProductRelated', 'Administrative_Duration']]

feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)

rfc = RandomForestClassifier(n_estimators=350, criterion='entropy', max_features='auto', random_state=1)
rfc.fit(X_train,Y_train)

Y_pred = rfc.predict(X_test)
conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Top 6')
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])
##Returns
##TP:  421
##TN:  2767
##FP:  338
##FN:  148


###TRY TOP 7
X = final_data[['PageValue', 'ExitRate', 'Administrative','ProductRelated_Duration', 'ProductRelated', 'Administrative_Duration', 'BounceRate']]

feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)

rfc = RandomForestClassifier(n_estimators=350, criterion='entropy', max_features='auto', random_state=1)
rfc.fit(X_train,Y_train)

Y_pred = rfc.predict(X_test)
conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Top 7')
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])
##Returns
##TP:  419
##TN:  2795
##FP:  310
##FN:  150


###TRY TOP 8
X = final_data[['PageValue', 'ExitRate', 'Administrative','ProductRelated_Duration', 'ProductRelated', 'Administrative_Duration', 'BounceRate', 'Month_Nov']]

feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)

rfc = RandomForestClassifier(n_estimators=350, criterion='entropy', max_features='auto', random_state=1)
rfc.fit(X_train,Y_train)

Y_pred = rfc.predict(X_test)
conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Top 8')
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])
##returns
##TP:  420
##TN:  2782
##FP:  323
##FN:  149

'''FALSE POSITIVES NOW INCREASING SO 7 GIVES A GOOD OPTION FOR MINIMISING FP @ 310 WITH A TP OF 419'''


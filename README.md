## Project Overview
This report provides a machine learning design for the best predictors of a transaction on an e-commerce website, along with accompanying analysis, using the given data set e-shop.csv. During each web session, a visitor may choose to browse any number of web pages, and each session may or may not result in a transaction. This project is divided into two parts.

#### Part 1: Random Forest 
A random forest classification model will be built in Python to predict whether a transaction will take place during a given web session or not. The python code and accompanying analysis will be included for submission.



#### Data Preparation
Before the data preparation, the important packages were installed, and the data was opened using pandas. 

###### Examining the data 
First the data was examined in its raw form using python: 
print(dataset.shape()) provided the number of rows and column: 12,245 & 14 respectively.
print(dataset.info()) provided information regarding the variable type:  integer, float, object, or Boolean.  We had a number of objects which required conversion to integers.  Note that Booleans did not need to be converted as TRUE = 1 and FALSE = 0 inherently.
print(dataset.describe()) provided basic statistical analysis for each of the variables.  There was a very large difference in the mean and standard deviation for all variables. Product Related_Duration had the highest mean and standard deviation (M = 1,199.00, SD = 1,916.73), while the Bounce Rate had the lowest (M = 0.02, SD = 0.05).

###### Converting Categorical variables into Numerical Variables
Categorical variable ‘VisitorType’ was convert into numerical variables using the ‘if’ statement in Python 3.7:  If the visitor was a new visitor, they were assigned a value of 1, otherwise they were given a value of 0.

###### Split Categorical Variable ‘Month’
‘Month’ was split into 10 columns, one for each of the months represented in the data and converted to integers.
If the transaction happened during that month, the value of 1 was assigned to that column, and 0 assigned to all other Month columns.  This increased the overall column count to 23 as our final_data variable shows.

###### Dividing dataset into label and feature sets
‘Transaction’ is our target variable, so this column is dropped, leaving behind 22 column variables in our dataframe ‘X’, with the original 12,245 rows. This is shown by the statement print(X.shape).

###### Normalise the data 
Due to the large variation in variables for mean and standard deviation, the data was normalised, so that each variable has a mean of 0 and a variance of 1. 
This is performed to ensure that there is equal importance placed on all the features. If this is not performed, then the machine learning model would assume that the ‘Product Related_Duration’ was of more significance than the other variables – simply because of its higher values.

###### Dividing dataset into training and test sets
The dataset was split into training and test sets. Training set was taken as 70%, while test set was 30% of the dataset (X_train = 8571, 22. X_test = 3674, 22). This is a typical standard split for data analysis.

###### Balance the data 
Due to the imbalance of the classes within the training set data (False = 7248 (not transaction) Majority class, True = 1323 (transaction) Minority class), the data had to be balanced.  If the data was not balanced, then the model built would be good at predicting the majority, but not the minority class. 
It was decided to oversample due to the small number of minority class, to bring the number of TRUE samples up to the level of FALSE samples – this creates artificial samples that are similar to existing ones, and inserts them into the existing samples in the minority class. Balancing was conducted on the training data, not the test set. This is because we want our algorithm to learn as much as it can from the training set, to be used on the test data – as this would almost always be imbalanced as in the real world.
If under sampling had been undertaken instead, this would have led to a smaller number of the majority class, so less for the algorithm to learn from.
The results of the oversampling show that each class is now balanced: False  = 7248, True = 7248.

###### Model Evaluation Strategy
During our evaluation, the decision was taken to focus the model on minimising false positives, the logic for this is as follows:
1.	The aim of the exercise is to create a predictive model for an e-commerce company to, as accurately as possible, predict future transactions based on previous successful transactions using the provided dataset. Due to the nature of e-commerce the dataset is always going to be imbalanced towards negative transactions (you will always have more visitors than customers) Based on this fact it would be logical to focus on actual transactions and base the model on confirmed true values.
2.	By focusing on reducing the amount of false positives it is possible that there may be an increased number of false negatives, this in our opinion is an acceptable error as this would relate to instances where a transaction did actually occur and was not predicted by the model, this in fact would have a positive impact on the company’s revenue and would then be recorded and feed back into the dataset and improve model accuracy overall.
3.	Focusing on classification accuracy was not a viable option based on the imbalanced nature of the data and the logical assumption the data will remain to be imbalanced in the future.
4.	It is our opinion that focusing the model to reduce false positives will give the most accurate outcome and will be the best predictor of future actual transactions. 

### Part 1 – Random Forestation - Model Building and Testing

###### Hyperparameter tuning
A range of ‘n_estimators’ was run [200, 250, 300, 350, 400, 450] to find the optimal number of decision trees, using scoring = ‘precision’, to minimise False Positives. Cross validation was also set to 5, to evaluate the model on a number of test sets to ensure that it consistently behaves the same way.  It divides the dataset into 5 equal folds parts, 1 will become the test set, 4 parts become the training set, then it rolls again - making sure that each fold becomes a test set while the others are training sets.  If this is consistent, it implies it is a good model.  It will do this for each maximum depth of decision trees listed as ‘n_estimators’.  The maximum number of decision trees which provides the most accurate model result, was 350. Since this falls mid-range of the ‘n_estimators’ set, there is no need for further tuning.

###### Performance testing for model built using all features
Using 350 trees, the following Features Importance List was generated, using all features, along with the results from the confusion matrix:
  
The figures for True Positive (TP), True Negative (TN), False Positive (FP) and False Negative (FN) are shown in the table below – along with the results of a range of Top Features, which will be discussed later. The table also shows model performance results. Please see Appendix 1 – Definitions, for definitions and formulae. 
 
As the table illustrates, the Precision rate for all features is 58.5%, a True Positive rate of 74.3% and an F1 score of 65.5%, which combines the True Positive Rate and Precision of a model.
These figures are not terribly high and suggests our model will not be as successful predicting transactions (TRUE), as compared to no-transaction (FALSE). Our model is very strong at predicting the True Negatives at 90.3%, but its ability to correctly identify an item as positive out of all items identified as positive is not strong (58.5%). However, it is better at correctly identifying an item as positive out of the total actual positives (74.3%). Overall, our model has a model accuracy rate of 87.9%. If we wanted to be able to predict any class, this model would be strong as it means that it would accurately predict 87.9% of the time, but since we want to accurately predict successful transactions (TRUE) this accuracy rate is misleading. A better measure of its overall performance would be the Matthew Correlation Coefficient Score (MCC) which:
...is a performance score that takes into account true and false positives, as well as true and false negatives. This score gives a good evaluation for imbalanced dataset.
Coinmonks, 2018
The MCC score for all features is 58.9%, which is not strong.
Performance testing for model built using a subset of significant features
In order to avoid overfitting, subsets of the features with greatest importance was run for top 5, 6, 7, 8, and the confusion matrices compared. We stopped at 8 features as the FP rate increased.
The above table shows the results and model performance for the top features.

###### Identifying Best Model
The best classification model is using the top 7 features, with 350 decision trees, using a cross validation of 5. The features and parameters are PageValue; ExitRate; Administrative; ProductRelated_Duration; ProductRelated; Administrative_Duration; and BounceRate. 
The above table shows the performance on the test set which returned Precision and True Positive rates of 57.5% and 73.6% respectively. It strongly predicts the majority class, as expected with an imbalanced data set, but given its overall objective is to predict Positives, the measure of F1 and MCC give a better assessment of its strength - 64.6% and 57.7% respectively, which is not particularly strong. 

### Recommendations
Firstly, we must note the limitations of this predictive model as discussed above. However, based on the predictive power of using the top 7 features in our model, we would make the following recommendations for increasing the number of transactions on the website:
Website traffic should be optimised such that external links, via google searches and Search Engine Optimization, are directed to the pages with high Page Values – as Page Values is the key driver in generating transactions. Additionally, all of e-shops website pages should include a hypertext link to drive traffic to these pages.  E-shop may also want to consider paid placement to further increase their traffic to these pages. Further analysis will be required to ascertain the revenue generated by visits to these pages, as some may generate low revenue sales, and others may generate higher revenue sales.
E-shop should also undertake a review of their pages which have a high exit rate as these pages are resulting in a customer leaving the website without making a purchase. The design of these pages, including the user interface should be reviewed to understand if there are elements of the page which are causing frustration or boredom for customers. A survey of customers and prospective customers may be advisable, to understand the reasons for visitors leaving these pages. E-shop should seek to redesign, add links driving traffic onto pages with high page values, or eliminate them if they add little value and are not legally required.  

Pages classified as type ‘Administrative’ are the next determinant of transactions. The original mean of the raw data set was 2.32 administrative pages visited per visit [print(dataset.Administrative.describe())]. 
A brief analysis of the csv file, filtered by FALSE or TRUE transactions, shows the average (mean) of the number of pages visited by each category:  2.12 FALSE, and 3.4 TRUE. This shows that more administrative pages were visited on average when a successful transaction occurred. This is logical, as it would require a customer to visit those pages in order to check account details/delivery addresses etc. Similarly, the average amount of time spent on those pages, as stated in the 6th determinant of transactions, ‘Administrative Duration’, is higher where the visit has resulted in a transaction (overall average was 81, average time where transaction was the result was 120, and for no transaction 74).  In the raw data set the maximum time spent on administrative pages was 3,399. The maximum time spent by customers who made a transaction was 2,087. Another brief analysis of the data set in excel shows that only 7 visits in excess of 2,087 time spent, did not result in a transaction. So this is a small proportion of customers and seem to be outliers rather than an issue with the pages themselves causing the customers to leave in frustration. 
 
As it generally appears that customers visit more administrative pages and spend longer on them, when making a purchase there may be an opportunity here for e-store to investigate what is/is not working for the customer by way of a survey. Also e-store should conduct a review the admin page to ensure they are as user-friendly as possible and to ensure the customer has a positive experience that will increase the likelihood of a repeat transaction.
The duration of visits to Product Related pages are the next important feature, with the average time being 1,199. The average for successful transactions was 1,881 and for no transaction 1,074. Similarly, there were more of these page types visited, on average by visits resulting in transactions. It is important for e-shop to ensure that their product pages are engaging as the more engaged a customer is, the more likely they are to complete a transaction.
Lastly, e-shop should analyse the pages with the highest Bounce Rate to identify the reasons for visitors entering the site and then leaving that specific page without visiting other pages. It may be that these pages are visited via external links which are not well optimised and are driving traffic to the site, but the site is not what the searcher had in mind. Alternatively, it could be that these pages are not engaging, even to potentially interested visitors. The reasons for the lack of engagement requires investigation in order to take appropriate action. 

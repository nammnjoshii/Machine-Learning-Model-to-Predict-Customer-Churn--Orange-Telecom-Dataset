
# Machine Learning Model to Predict Customer Churn - Orange Telecom Dataset

### Author : Nammn Joshii

### Project Description

The challenging industry dynamics in telecom industry bring us to discuss the top priority of any telecom provider which is to manage their customer base and reduce churn. Using the Orange Telecom data, the project aspire to develop a robust model which identifies the key variables that lead to churn and alert a telecom provider which customer might unsubscribe their services. 

### File

In this project, following files are included:

a. churn-bigml-80.csv: original training data set downloaded from Kaggle (https://www.kaggle.com/mnassrib/telecom-churn-datasets). The data consists of customer activity data, such as the number and length of day and night calls and the number of customer service calls, along with a churn label specifying whether a customer cancelled the subscription. 
Our endevour is to predict customers’ future decisions based on their past behavior.
b. churn-bigml-20.csv: original testing data set downloaded from Kaggle (https://www.kaggle.com/mnassrib/telecom-churn-datasets)
c. Orange Telecom Churn project.ipynb: python coding file
d. Final report_Orange Telecom Churn.pdf: final report

### Python library installation

Required libraries : python scikit-learn, NumPy, pandas

Required scikit-learn modules : 

a. sklearn.model_selection - cross_validate, train_test_split, GridSearchCV <br /> 
b. sklearn.tree - DecisionTreeClassifier <br /> 
c. sklearn.preprocessing - OneHotEncoder, StandardScaler <br /> 
d. sklearn.compose - ColumnTransformer <br /> 
e. sklearn.naive_bayes - MultinomialNB <br /> 
f. sklearn.feature_extraction.text - CountVectorizer <br /> 
g. sklearn.feature_selection - SelectKBest, chi2, mutual_info_classif, SelectKBest, f_regression <br /> 
h. sklearn.metrics - mutual_info_score, confusion_matrix <br /> 
i. sklearn.linear_model - LogisticRegression <br /> 
k. sklearn.neighbors - KNeighborsClassifier <br /> 
l. sklearn - preprocessing <br /> 
m. sklearn.ensemble - RandomForestClassifier <br /> 
n. sklearn.pipeline - Pipeline <br /> 
o. sklearn.datasets - make_classification <br /> 


### Approach

The original data set from Kaggle has already been split into two csv files representing training set (80%) and testing set(20%) respectively. Ho
ver, in order to avoid overfitting models due to inappropriate splitting of training and testing set, re-did the splitting by ourselves using scikit learn function. The full dataset is now splitted into 80% as training set and 20% as testing set in the python file.

For model selection only use training set to learn the model. For validation purpose, decided to split the training data into 5-folds for cross validation. Following five models are evaluated: 

a. Naive bayes classifier <br /> 
b. Decision tree classifier <br /> 
c. Logistic regression <br /> 
d. kNN classifier <br /> 
e. Random forest classifier <br />   

The final model selected is random forest classifier with max depth of 50 and 120 estimators using “churn” label as predictor variable and all variables except for “voicemail plan” and “area code” as features.


### Usage

Telecom companies can utilise our model and results to identify potential loss of current customers and take immediate action to prevent such loss. Meanwhile, by estimating the overall churn rate, Telecom companies can better position themselves in the market and understand its advantages/disadvantages compared to competitors. Finally, the top features can be used to improve their plans and services for better customer retention.


### License

Data files © Original Authors

### Further Developments 

Orange Telecom’s data lacks market data and competitors’ actions. It might be case that sometimes customers cancel their subscription just because some other telecom companies provide better plans or have a new customer promotion. 
It would be interesting to see the integration of competitors data with this data set to better assess the impacts of competitors’ actions. 



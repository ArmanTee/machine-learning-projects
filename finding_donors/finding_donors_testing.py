# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
import os
os.getcwd()
data= pd.read_csv('finding_donors/census.csv')

# Import supplementary visualization code visuals.py
import finding_donors.visuals as vs

# Pretty display for notebooks
%matplotlib inline


# Success - Display the first record
display(data.head(n=1))
n_records = len(data)
## Some stats on data
is_more_than_50k=(data['income'] == '>50K')
n_greater_50k= len(data[is_more_than_50k])
# TODO: Number of records where individual's income is at most $50,000
is_less_than_50k=(data['income'] == '<=50K')
n_at_most_50k = len(data[is_less_than_50k])

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = 100*n_greater_50k/(n_greater_50k+n_at_most_50k)

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))
'''
Featureset Exploration

age: continuous.

workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov,
 State-gov, Without-pay, Never-worked.

education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm,
 Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.

marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
 Married-spouse-absent, Married-AF-spouse.

occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial,
Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical,
Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.

relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: Black, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other.

sex: Female, Male.

capital-gain: continuous.

capital-loss: continuous.

hours-per-week: continuous.

native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany,
 Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran,
 Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland,
 France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary,
 Guatemala, Nicaragua, Scotland, Thailand,  Yugoslavia, El-Salvador, Trinadad&Tobago,
 Peru, Hong, Holand-Netherlands.
 '''

#### HISTOGRAM OF INCOME DATA ###
 # Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)
data['capital-loss']
# Visualize skewed continuous features of original data
vs.distribution(data)
from matplotlib import pyplot as plt
### LOG TRANSFORM THE DATA TO HELP REMOVE SKEW ###
# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed = True)



### MINIMAX SCALING ####
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))

# TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
features_final = pd.get_dummies(features_log_minmax_transform)

# TODO: Encode the 'income_raw' data to numerical values
income = (pd.get_dummies(data['income']))['>50K']

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
# print encoded

# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                    income,
                                                    test_size = 0.2,
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
income
'''
TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data
encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case
'''
# TODO: Calculate accuracy, precision and recall
TP= np.sum(income)
totalPos = len(income)
FP = totalPos-TP
TN = 0
FN = 0
accuracy = TP/totalPos
recall = TP/(TP+FN)
precision = TP/(TP+FP)

# TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
def fbetaScore(beta,recall,precision):
    return ((1 + beta**2)*(precision*recall))/((beta**2*precision)+recall)
fscore = fbetaScore(0.5,recall,precision)

# Print the results
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))

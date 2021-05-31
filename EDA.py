''' Exploratory data analysis '''


import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns





# read dataset
try:
    dataset = pd.read_csv('creditcard.csv')
except:
    print('Unable to find data on folder, please download data from the available link ')

# frst five rows
first_5_columns = dataset.head()
print(first_5_columns)

print('######################################')





# data types of values in dataset
print('dtypes and memory usage of dataset: ')
print(dataset.info())

print('')
print('######################################')






# missing values
missing_values = dataset.isnull().sum() # no missing values

print('missing values: ')
print(missing_values)

print('')
print('######################################')




# split data into fraud class dataset and normal class dataset
fraud_trans = dataset[dataset['Class'] == 1]
normal_trans = dataset[dataset['Class'] == 0]




# count target and plot the counts
fraud = dataset['Class'].value_counts()[1]
normal_transactions = dataset['Class'].value_counts()[0]

print(f'total count of fraud transactions is : {fraud}')
print('')
print(f'total count of normal transactions is : {normal_transactions}')
print('')
print('######################################')






# amount info
amount_info_fraud = fraud_trans['Amount'].describe()
amount_info_normal = normal_trans['Amount'].describe()
print(f'amount column info in fraud class:\n{amount_info_fraud}')
print('')
print(f'amount column info in normal class:\n{amount_info_normal}')
print('')
print('######################################')






# average of transactions amount for fraud and normal
group_class_by_amount = dataset[['Class', 'Amount']].groupby(['Class']).mean()
print(f'average of transactions amount for normal class is {group_class_by_amount.iloc[0][0]:.2f}$')
print(f'average of transactions amount for fraud classes is {group_class_by_amount.iloc[1][0]:.2f}$')

print('')
print('######################################')






# plot count of classes
target_counts = sns.countplot(x="Class", data=dataset)


# drop time column 
dataset = dataset.drop('Time', axis=1)


dataset.to_csv('new_creditcard.csv')




























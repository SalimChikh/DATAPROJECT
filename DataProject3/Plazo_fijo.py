#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:45:59 2020

@author: salim
"""
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
from sklearn.cluster import KMeans
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #para dibujos
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split

os.chdir('/Users/salim/Desktop/EDEM/machine_learning/HACKATHON')
os.getcwd() #despues de cambiar de directorio verificamos si se ha hecho bien
df = pd.read_csv ("BIG_BANK_prestamo.csv", sep=',', decimal ='.')

data = pd.read_csv ("BIG_BANK_prestamo.csv", sep=',', decimal ='.')
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

# Display up to 60 columns of a dataframe
pd.set_option('display.max_columns', 60)

# Set default font size
plt.rcParams['font.size'] = 24

# Display top of dataframe
data.head()
data.shape
data.describe()


#limpiar 

df.drop(labels=['Loan ID', 'Customer ID'], axis=1, inplace=True)
#comptar el numero de nulos  
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

missing_values_table(data)

data.drop(columns = 'Months since last delinquent', axis=1, inplace=True)
# Here I can see that the last 514 observations are NaN values.

data[data['Years of Credit History'].isnull() == True]

data.drop(data.tail(514).index, inplace=True) # drop last 514 rows
missing_values_table(data)

# As the number of missing values is so low in the 'Maximum Open Credit' I will drop them.

for i in data['Maximum Open Credit'][data['Maximum Open Credit'].isnull() == True].index:
    data.drop(labels=i, inplace=True)
missing_values_table(data)


# As the number of missing values is so low in the 'Tax Liens' I will drop them.

for i in data['Tax Liens'][data['Tax Liens'].isnull() == True].index:
    data.drop(labels=i, inplace=True)
missing_values_table(data)

# As the number of missing values is so low in the 'Bankruptcies' I will drop them.

for i in data['Bankruptcies'][data['Bankruptcies'].isnull() == True].index:
    data.drop(labels=i, inplace=True)
missing_values_table(data)

######
# Now I will use the 'mean' technique to fill the NaN values.

for i in data['Years in current job'][data['Years in current job'].isnull() == True].index:
    data.drop(labels=i, inplace=True)
data['Years in current job']= data['Years in current job'].fillna(value='10+ years')

data.fillna(data.mean(), inplace=True)
missing_values_table(data)

# The feature 'Years in current job' didn't fill because has categorical values.


# I will figure out what value is more present in this feature.
#data=data.iloc[:1000]
data2=data.iloc[:5000]
plt.figure(figsize=(20,8))

sns.countplot(data['Years in current job'])



# No missing values anymore.
sns.pairplot(data)

# Correlations between Features and Target

# Find all correlations and sort 
correlations_data = data.corr()['Credit Score'].sort_values(ascending=False)
data.detail()
data.dtypes

# Print the correlations
print(correlations_data.tail)

# Select the categorical columns
categorical_subset = data[['Term', 'Years in current job', 'Home Ownership', 'Purpose']]

# One hot encode
categorical_subset = pd.get_dummies(categorical_subset)

# Join the dataframe in credit_train
# Make sure to use axis = 1 to perform a column bind
# First I will drop the 'old' categorical datas and after I will join the 'new' one.

data.drop(labels=['Term', 'Years in current job', 'Home Ownership', 'Purpose'], axis=1, inplace=True)
data = pd.concat([data, categorical_subset], axis = 1)

# #  Remove Collinear Features

def remove_collinear_features(x, threshold):

    y = x['Loan Status']
    x = x.drop(columns = ['Loan Status'])
    
    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            
            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns = drops)
    
    # Add the score back in to the data
    x['Loan Status'] = y
               
    return x


# Remove the collinear features above a specified correlation coefficient
data = remove_collinear_features(data, 0.6);

data.shape

# # # Split Into Training and Testing Sets

# Separate out the features and targets
features = data.drop(columns='Loan Status')
targets = pd.DataFrame(data['Loan Status'])


# Split into 80% training and 20% testing set
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# # Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y_train = LabelEncoder()
y_train = labelencoder_y_train.fit_transform(y_train)
labelencoder_y_test = LabelEncoder()
y_test = labelencoder_y_test.fit_transform(y_test)


pd.csv_write ("BIG_BANK_prestamo_filtrado.csv", sep=',', decimal ='.')

data.to_csv('out.csv')



# implemented from
# shttps://medium.com/datadriveninvestor/a-simple-guide-to-creating-predictive-models-in-python-part-1-8e3ddc3d7008

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# loads data
df = pd.read_csv('Churn_Modelling.csv')

# removing the irrelevant columns
cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
df = df.drop(columns=cols_to_drop, axis=1)

# first five rows of data frame after removing columns
# just print the page
# df.info()
# df.head()

# The reason for copy dataframe this will be explained in
# the 'Artificial Neural Network' part in 'Modelling' which is in Part-2.
deep_df = df.copy(deep=True)


#
# listing non-continuous numerical value to find outlier
#


# like non-same datatype
numerical_columns = [
    col for col in df.columns
    if (df[col].dtype == 'int64' or df[col].dtype == 'float64') and col != 'Exited'
]

# just for check: at lease 50% data equal to validation
print(
    'just for check: at lease 50% data equal to validation \n',
    df[numerical_columns].describe().loc[['min', 'max', 'mean', '50%'], :]
)

# The above small analysis shows that the person is actually a 45 years old male
# and already has a credit card with high credit score and a balance of almost 123 K.
# But he has an estimated salary of only 11.58 which is pretty weird.
# Maybe it is just an error in Data collection or maybe he just lost his job or possibly got retired.
# We can consider it as an outlier and delete that row from the Data Frame
# but it is a judgement you have to make as a Data Scientist/Analyst.
print(
    'get minimal estimate salary \n',
    df[df['EstimatedSalary'] == df['EstimatedSalary'].min()]
)

percentages = []

# 1. decide if the 'Gender' column is relevant.

for gen in list(df['Gender'].unique()):
    p = round((df['Exited'][df['Gender'] == gen].value_counts()[1] / df['Exited'][
        df['Gender'] == gen].value_counts().sum()) * 100, 2)

    percentages.append(p)
    print(gen, '(% to exit) : ', p)

# 2. Visualizing the results

plt.bar(0, percentages[0])
plt.bar(1, percentages[1])
plt.xticks((0, 1), ('Female', 'Male'))
plt.xlabel('Gender')
plt.ylabel('Percentage')
plt.title('Percentage of gender to Exit')
plt.show()

percentages = []

print('\n')

# 1. decide if the 'Geography' column is relevant by doing similar analysis as above

for country in list(df['Geography'].unique()):
    p = round((df['Exited'][df['Geography'] == country].value_counts()[1] / df['Exited'][
        df['Geography'] == country].value_counts().sum()) * 100, 2)

    percentages.append(p)
    print(country, '(% to exit) : ', p)

# 2. Visualizing the results

for i in range(len(percentages)):
    plt.bar(i, percentages[i])
plt.xticks((0, 1, 2), ('France', 'Spain', 'Germany'))
plt.xlabel('Country')
plt.ylabel('Percentage')
plt.title('Percentage of Country to Exit')
plt.show()

# plotting a histogram of the 'Age' column

plt.hist(df['Age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')

# the age of customers(y-axis) who quit the bank ('Exited' = 1)

plt.scatter(x=range(len(list(df['Age'][df['Exited'] == 0]))), y=df['Age'][df['Exited'] == 0], s=1)
plt.ylabel('Age')
plt.xlabel('People (rows)')
plt.title('People who did not Exit (Exited = 0)')
plt.show()

# the 'percentage' of people who quit ('Exited' = 1) in each age group

plt.scatter(x=range(len(list(df['Age'][df['Exited'] == 1]))), y=df['Age'][df['Exited'] == 1], s=1)
plt.ylabel('Age')
plt.xlabel('People (rows)')
plt.title('People who Exited (Exited = 1)')
plt.show()


# bucketizing the age column and using 'groupby' to create groups for each age group

age_bucket = df.groupby(pd.cut(df['Age'], bins=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))

age_bucket = ((age_bucket.sum()['Exited'] / age_bucket.size()) * 100).astype(float)

x = [str(i) + '-' + str(i + 10) for i in range(10, 91, 10)]
plt.plot(x, age_bucket.values)
plt.xlabel('Age Group')
plt.ylabel('Percentage exited')
plt.title('Percentage of people in different Age Groups that exited')
plt.show()

# replace the 'Age' column with the bucketized column

df['Age'] = pd.cut(df['Age'], bins=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
df.head()

# change Age into categorical column

df = pd.get_dummies(df)

# remove (any) one of the encoded column each from the parent column.

df = df.drop(columns=['Geography_France', 'Gender_Female'], axis=1)

# remove the bucketized dummy Age column 'Age_(90,100]' by selecting all the columns except the last one

df = df.iloc[:, :-1]

df.to_csv('Clean_Data.csv')


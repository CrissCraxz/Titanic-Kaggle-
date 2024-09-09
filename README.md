#  Titanic Survival Prediction


This project uses a dataset from the Titanic to predict whether a passenger survived or not. The training dataset is analyzed and preprocessed, and a RandomForestClassifier is trained to make predictions on the test dataset.
Requirements

To run this project, you'll need the following Python libraries installed:

    pandas
    matplotlib
    seaborn
    scikit-learn

You can install them with:

bash

pip install pandas matplotlib seaborn scikit-learn

Project Overview

The main steps of the project are:

    Loading the Data: Titanic datasets for training and testing are loaded using pandas.
    Data Exploration: We analyze the dataset for missing values, duplicated data, and general statistics.
    Data Visualization: Using seaborn, we create visualizations to gain insights into the dataset.
    Data Preprocessing: Irrelevant columns are dropped, and categorical variables are encoded.
    Model Training: A RandomForestClassifier is trained using the preprocessed data.
    Making Predictions: The trained model is used to predict survival on the test dataset, and results are exported to a CSV file.

Steps
1. Loading the Data

python

import pandas as pd

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

2. Data Exploration

Explore the dataset with some basic operations:

python

# Sample of the dataset
df_train.sample(5)

# Shape of the dataset
df_train.shape

# Dataset info
df_train.info()

# Data types
df_train.dtypes

# Check for duplicates
df_train.duplicated().sum()

# Check for missing values
df_train.isnull().sum().sort_values(ascending=False)

# Count of unique values
df_train.nunique()

3. Analyzing Categories and Numeric Columns

Check the categories and numerical columns with limited values:

python

# Categorical columns
column_categories = df_train.select_dtypes(include=['object']).columns
for columns in column_categories:
    if df_train[columns].nunique() <= 10:
        print(f"{columns}: {df_train[columns].unique()}")
        
# Numeric columns
column_numerics = df_train.select_dtypes(include=['int64', 'float64']).columns
for columns in column_numerics:
    if df_train[columns].nunique() <= 10:
        print(f"{columns}: {df_train[columns].unique()}")

4. Data Visualization

Create visualizations to explore survival rates and other features:

python

import seaborn as sns
import matplotlib.pyplot as plt

# Count of survivors
sns.countplot(x='Survived', data=df_train)
plt.show()

# Survival rate by sex
sns.barplot(x='Sex', y='Survived', data=df_train)
plt.show()

5. Data Preprocessing

Drop irrelevant columns that won’t be used in the model:

python

df_train = df_train.drop(columns=['Cabin', 'Fare', 'Ticket', 'Name'])
df_test = df_test.drop(columns=['Cabin', 'Fare', 'Ticket', 'Name'])

6. Encoding Categorical Variables

Encode the categorical variables using OrdinalEncoder:

python

from sklearn.preprocessing import OrdinalEncoder

X = df_train.drop(['Survived'], axis=1)
y = df_train.Survived

# Identify categorical columns
s = (X.dtypes == 'object')
object_cols = list(s[s].index)

# Encode the categorical columns
ordinal_encoder = OrdinalEncoder()
X[object_cols] = ordinal_encoder.fit_transform(X[object_cols])

7. Imputing Missing Values

Handle missing values using SimpleImputer:

python

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
x_transformed = pd.DataFrame(imputer.fit_transform(X))
x_transformed.columns = X.columns

# Verify no missing values remain
x_transformed.isnull().sum()

8. Model Training

Train a RandomForestClassifier:

python

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x_transformed, y)

9. Making Predictions

Preprocess the test data and make predictions:

python

# Transform the test set categorical columns
df_test[object_cols] = ordinal_encoder.fit_transform(df_test[object_cols])

# Impute missing values in the test set
df_test_transformed = pd.DataFrame(imputer.transform(df_test))
df_test_transformed.columns = df_test.columns

# Make predictions
predictions = model.predict(df_test_transformed)

10. Exporting Results

Save the predictions to a CSV file:

python

output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

Conclusion

This project demonstrates a basic workflow of data preprocessing, visualization, model training, and prediction using Python libraries like pandas, seaborn, and scikit-learn.

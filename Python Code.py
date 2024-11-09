# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:35:58 2024

@author: zohoo
"""


#Import libraries

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Exploratory Data Analysis (EDA)

#load data
Data_Set=pd.read_csv(r"C:\Users\zohoo\OneDrive\Desktop\Hospital Readmission\Hospital-Readmission-\Data\hospital_readmissions.csv")

#1- Data Overview:
print(Data_Set.shape)
print(Data_Set.dtypes)
print(Data_Set.isnull().sum())



#2- Descriptive statistics

print(r"\Numerical colmns summary:")
print(Data_Set.describe())
print(r"\nCategorical columns summary:")
for col in Data_Set.select_dtypes(include=('object')):
    print(r"\n{column}:")
    print(Data_Set[col].value_counts())
      

#3- Distribution analysis
numerical_columns=Data_Set.select_dtypes(include=[np.number]).columns
for col in numerical_columns:
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    Data_Set[col].hist()
    plt.title(f'Histogram of {col}')
    plt.subplot(122)
    stats.probplot(Data_Set[col], dist="norm",plot=plt)
    plt.title(f'Q-Q plot of {col}')
    plt.tight_layout()
    plt.show()
    
#4- Correlation analysis

corr_matrix=Data_Set[numerical_columns].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix,annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
    
    
#5- Target variable analysis

if 'readmitted' in Data_Set.columns:
    plt.figure(figsize=(8,6))
    Data_Set['readmitted'].value_counts().plot(kind='bar')
    plt.show()
else:
    print("No readmitted column in dataframe")
    


#6- Feature relationship

if 'readmitted' in Data_Set.columns:
    for col in numerical_columns:
        if col!='readmitted':
            plt.figure(figsize=(10,6))
            sns.boxplot(x='readmitted',y=col,data=Data_Set)
            plt.title(f'{col}vs readmission')
            plt.xticks(rotation=45)
            plt.show()
        else:
            print("No column found")
            
            
categorical_column=Data_Set.select_dtypes(include=['object']).columns
for col in categorical_column:
    print(col)
    
    plt.figure(figsize=(10,6)) 
    sns.countplot(x=col, hue='readmitted',data=Data_Set)
    plt.title(f'{col} vs Readmission')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("No readmitted column")
       
            
            
    
#7- Multivariate analysis
sns.pairplot(Data_Set[numerical_columns])
plt.show()



#Regression model



# 1. Define numerical and categorical columns
numerical_cols = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 
                  'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency']

categorical_cols = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
                    'glucose_test', 'A1Ctest', 'change', 'diabetes_med']





# 3. Preprocessing step: Scaling numerical and encoding categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),          # Scaling numerical data
        ('cat', OneHotEncoder(drop='first'), categorical_cols) # One-hot encoding categorical data
    ]
)


# Define the pipeline before calling fit()
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Preprocessing step
    ('classifier', LogisticRegression(max_iter=1000))  # Logistic regression model
])


# 5. Split the dataset
X = Data_Set.drop('readmitted', axis=1)
y = Data_Set['readmitted']

# Ensure your target variable is binary (0, 1) for logistic regression
# If 'readmitted' is categorical (e.g., 'yes', 'no'), map it to binary values
y = y.map({'yes': 1, 'no': 0}) if y.dtype == 'object' else y

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train the model using the pipeline
pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)


# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

















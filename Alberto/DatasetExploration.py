import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

################
# IMPORTING DATA
################

# Import train data
train_data = pd.read_csv('../diabetes_train.csv')
# Import test data
test_data = pd.read_csv('../diabetes_test.csv')

#######################
# EXPLORING THE DATASET
#######################

# Check the first 5 rows of the train data
print("First rows: \n", train_data.head())
# Check the columns of the train data
print("Labels: ",train_data.columns)
# Check the data types of the train data
print("Data types: \n",train_data.dtypes, "\n")

# Check the unique values of the train data
#print(train_data.nunique())
# Check the unique values of the test data
#print(test_data.nunique())

# Check the shape of the test and train data
print("Train data shape: ", train_data.shape)
print("Test data shape: ", test_data.shape, "\n")

# Check the null values of the train data   
print("Null values in train (for column): \n",train_data.isnull().sum(), "\n")
# Check the null values of the test data
print("Null values in test (for column): \n",test_data.isnull().sum(), "\n")

# Replace null values with the average for numerical columns in train data
for col in train_data.select_dtypes(include=np.number).columns:
    train_data[col].fillna(train_data[col].mean(), inplace=True)
# Replace null values with the average for numerical columns in test data
for col in test_data.select_dtypes(include=np.number).columns:
    test_data[col].fillna(test_data[col].mean(), inplace=True)



# Plot age vs Frequency (normalized)
plt.figure(figsize=(8, 5))
sns.histplot(data=train_data, x='age', stat='density', kde=True, color='orange')
plt.title("Normalized Histogram of Age (train)")
plt.xlabel("Age")
plt.ylabel("Density")
plt.show()

# Scatter plot of age on y-axis and smoking history on x-axis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_data, x='smoking_history', y='age')
plt.title("Scatter Plot of Age vs Smoking History (train)")
plt.xlabel("Smoking History")
plt.ylabel("Age")
plt.xticks(rotation=45)
plt.show()
# Plot histogram of smoking_history
plt.figure(figsize=(8, 5))
sns.histplot(data=test_data, x='smoking_history', kde=False, bins=10, color='green')
plt.title("Histogram of Smoking History (train)")
plt.xlabel("Smoking History")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

# Plot smoking history vs hypertension with percentage of hypertensive people
plt.figure(figsize=(10, 6))
hypertension_percentage = train_data.groupby('smoking_history')['hypertension'].mean() * 100
sns.barplot(x=hypertension_percentage.index, y=hypertension_percentage.values, palette="viridis")
plt.title("Percentage of Hypertensive People by Smoking History")
plt.xlabel("Smoking History")
plt.ylabel("Percentage of Hypertensive People")
plt.xticks(rotation=45)
plt.show()

# Scatter plot of age on y-axis and hypertension on x-axis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_data, x='hypertension', y='age')
plt.title("Scatter Plot of Age vs Hypertension (train)")
plt.xlabel("Hypertension")
plt.ylabel("Age")
plt.xticks([0, 1], labels=["No", "Yes"])
plt.show()

# Scatter plot of age on y-axis and heart_disease on x-axis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_data, x='heart_disease', y='age')
plt.title("Scatter Plot of Age vs Heart Disease (train)")
plt.xlabel("Heart Disease")
plt.ylabel("Age")
plt.xticks([0, 1], labels=["No", "Yes"])
plt.show()

# Bubble plot of BMI vs Age with Diabetes as size
plt.figure(figsize=(10, 6))
bubble_sizes = train_data['diabetes'] * 100  # Scale diabetes values for bubble size
sns.scatterplot(data=train_data, x='bmi', y='age', size=bubble_sizes, sizes=(20, 200), hue='diabetes', palette="coolwarm", alpha=0.6)
plt.title("Bubble Plot of BMI vs Age with Diabetes")
plt.xlabel("BMI")
plt.ylabel("Age")
plt.legend(title="Diabetes", loc='upper right')
plt.show()

# Plot BMI vs Frequency (normalized)
plt.figure(figsize=(8, 5))
sns.histplot(data=train_data, x='bmi', stat='density', kde=True, color='blue')
plt.title("Normalized Histogram of BMI (train)")
plt.xlabel("BMI")
plt.ylabel("Density")
plt.show()

# Plot histogram of diabetes
plt.figure(figsize=(8, 5))
sns.histplot(data=train_data, x='diabetes', kde=False, bins=2, color='purple')
plt.title("Histogram of Diabetes (train)")
plt.xlabel("Diabetes")
plt.ylabel("Frequency")
plt.xticks([0, 1], labels=["False", "True"])
plt.show()

# Plot smoking history vs diabetes with percentage of diabetic people
plt.figure(figsize=(10, 6))
diabetes_percentage = train_data.groupby('smoking_history')['diabetes'].mean() * 100
sns.barplot(x=diabetes_percentage.index, y=diabetes_percentage.values, palette="magma")
plt.title("Percentage of Diabetic People by Smoking History")
plt.xlabel("Smoking History")
plt.ylabel("Percentage of Diabetic People")
plt.xticks(rotation=45)
plt.show()

# Scatter plot of diabetes vs HbA1c levels
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_data, x='HbA1c_level', y='diabetes', hue='diabetes', palette="coolwarm", alpha=0.7)
plt.title("Scatter Plot of Diabetes vs HbA1c Levels")
plt.xlabel("HbA1c Level")
plt.ylabel("Diabetes")
plt.xticks(rotation=45)
plt.show()

# Scatter plot of diabetes vs glucose levels
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_data, x='blood_glucose_level', y='diabetes', hue='diabetes', palette="coolwarm", alpha=0.7)
plt.title("Scatter Plot of Diabetes vs Glucose Levels")
plt.xlabel("Glucose Level")
plt.ylabel("Diabetes")
plt.xticks(rotation=45)
plt.show()

# Scatter plot of diabetes vs insulin sensitivity
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_data, x='Insulin_Sensitivity_Est', y='diabetes', hue='diabetes', palette="coolwarm", alpha=0.7)
plt.title("Scatter Plot of Diabetes vs Insulin Sensitivity")
plt.xlabel("Insulin Sensitivity")
plt.ylabel("Diabetes")
plt.xticks(rotation=45)
plt.show()

# Plot diabetes vs BMI and Glucose interaction
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_data, x='BMI_Glucose_Interaction', y='blood_glucose_level', hue='diabetes', palette="coolwarm", alpha=0.7)
plt.title("Scatter Plot of BMI vs Glucose Level with Diabetes")
plt.xlabel("BMI")
plt.ylabel("Glucose Level")
plt.legend(title="Diabetes", loc='upper right')
plt.show()

#######################
# RESCALING THE DATASET
#######################

# Show rows with age <= 0
rows_with_age_leq_0 = train_data[train_data['age'] <= 0]
print("Rows with age <= 0: \n", rows_with_age_leq_0, "\n")
# Remove rows with age <= 0
train_data = train_data[train_data['age'] > 0]

# Check for values different than 'Male' or 'Female' in the gender column for train data
invalid_gender_values_train = train_data[~train_data['gender'].isin(['Male', 'Female'])]
print("Rows with invalid gender values in train data: \n", invalid_gender_values_train)
# Check for values different than 'Male' or 'Female' in the gender column for test data
invalid_gender_values_test = test_data[~test_data['gender'].isin(['Male', 'Female'])]
print("Rows with invalid gender values in test data: \n", invalid_gender_values_test)
# Assign 0 to male and 1 to female in the gender column for train data
train_data['gender'] = train_data['gender'].replace({'Male': 0, 'Female': 1})
# Assign 0 to male and 1 to female in the gender column for test data
test_data['gender'] = test_data['gender'].replace({'Male': 0, 'Female': 1})

# Replace 'ever' with 'never' and 'not current' with 'former' in smoking_history for train data
train_data['smoking_history'] = train_data['smoking_history'].replace({'ever': 'never', 'not current': 'former'})
# Replace 'ever' with 'never' and 'not current' with 'former' in smoking_history for test data
test_data['smoking_history'] = test_data['smoking_history'].replace({'ever': 'never', 'not current': 'former'})
# Plot histogram of smoking_history
plt.figure(figsize=(8, 5))
sns.histplot(data=train_data, x='smoking_history', kde=False, bins=10, color='green')
plt.title("Histogram of Smoking History (train, adjusted)")
plt.xlabel("Smoking History")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

# Show rows with BMI >= 80
rows_with_bmi_geq_80 = train_data[train_data['bmi'] >= 80]
print("Rows with BMI >= 80: \n", rows_with_bmi_geq_80, "\n")
# Remove rows with BMI >= 80
train_data = train_data[train_data['bmi'] < 80]


# Compute the covariance matrix
#cov_matrix = train_data.cov()

# Plot the covariance matrix
#plt.figure(figsize=(10, 8))
#sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="coolwarm")
#plt.title("Covariance Matrix of Train Data")
#plt.show()
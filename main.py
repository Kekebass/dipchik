import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import datetime
from datetime import datetime
# Configuration for graph width and layout
sns.set_theme(style='whitegrid')
palette='viridis'

# Warnings remove alerts
import warnings
warnings.filterwarnings("ignore")

# Set the display.max_columns option to None
pd.set_option('display.max_columns', None)

## Data 1
# Data train
train_df = pd.read_csv("train.csv")

# Data test
test_df = pd.read_csv("test.csv")

# Data 2
df = pd.read_csv("htrain.csv")

train_df.head()

# Viewing 5 latest data
train_df.tail()

# Info data
train_df.info()

# Type dados
train_df.dtypes

# Viewing rows and columns
train_df.shape

# Exploratory data analysis (EDA)
print("\nDescriptive statistics of the training set:")
train_df.describe().T

# Analysis of categorical and numerical variables
categorical_features = train_df.select_dtypes(include=['object']).columns
numerical_features = train_df.select_dtypes(include=[np.number]).columns

print("\nCategorical Variables:", categorical_features)
print("Numeric Variables:", numerical_features)

# Analysis of categorical variables
for col in categorical_features:
    print(f"\nDistribution of categorical variable {col}:")
    print(train_df[col].value_counts())

    # Analysis of target variable 'Target'
    print("\nDistribution of target variable 'Target':")
    print(train_df['Response'].value_counts())
    plt.figure(figsize=(10, 6))
    sns.countplot(data=train_df, x='Response')
    plt.title("Distribution of Target Variable 'Target'")
    plt.grid(False)
    plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Vehicle_Age', y='Annual_Premium', hue='Response', data=train_df)
plt.title('Boxplot of Annual Premium by Vehicle Age and Response')
plt.grid(False)
plt.show()

# Group data by gender and calculate the total annual premium for each group
total_premium_by_gender = train_df.groupby('Gender')['Annual_Premium'].sum().reset_index()

# View total prize by gender
plt.figure(figsize=(10, 6))
sns.barplot(x='Gender', y='Annual_Premium', data=total_premium_by_gender)
plt.title('Total Prize for Sex')
plt.xlabel('Sex')
plt.ylabel('Annual Premium Total')
plt.grid(False)
plt.show()

# Create age ranges
train_df['Age_Bucket'] = pd.cut(train_df['Age'], bins=[18, 25, 35, 50, np.inf], labels=['18-25', '26-35', ' 36-50', '51+'])

# Group data by age group and gender, and calculate the average annual premium for each group
average_premium_by_age_gender = train_df.groupby(['Age_Bucket', 'Gender'])['Annual_Premium'].mean().reset_index()

# View the average annual premium by age group and gender
plt.figure(figsize=(20, 10))
sns.barplot(x='Age_Bucket', y='Annual_Premium', hue='Gender', data=average_premium_by_age_gender)
plt.title('Average Annual Award by Age Group and Sex')
plt.xlabel('Age Range')
plt.ylabel('Average Annual Premium')
plt.legend(title='Sex')
plt.grid(False)
plt.show()

# Group data by gender and previous insurance status, and calculate the average annual premium for each group
average_premium_by_gender_insured = train_df.groupby(['Gender', 'Previously_Insured'])['Annual_Premium'].mean().reset_index()

# Transform the 'Previously_Insured' variable into a more readable category
average_premium_by_gender_insured['Previously_Insured'] = average_premium_by_gender_insured['Previously_Insured'].map({0: 'No', 1: 'Yes'})

# View average annual premium by gender and previous insurance status
plt.figure(figsize=(10, 6))
sns.barplot(x='Previously_Insured', y='Annual_Premium', hue='Gender', data=average_premium_by_gender_insured)
plt.title('Average Annual Premium by Previous Insurance Status and Gender')
plt.xlabel('Previous Insurance')
plt.ylabel('Average Annual Premium')
plt.legend(title='Sex')
plt.grid(False)
plt.show()

#Categorical variables to iterate
categorical_variables = ['Gender', 'Vehicle_Damage', 'Vehicle_Age', 'Response']

# Figure size
plt.figure(figsize=(15, 10))

# Loop over categorical variables
for i, var in enumerate(categorical_variables, 1):
 plt.subplot(2, 2, i) # Subplots 2x2
 sns.boxplot(data=train_df, x=var, y='Annual_Premium', palette='viridis')
 plt.title(f'Annual Award for {var}')
 plt.xlabel(var)
 plt.ylabel('Annual Award')
 plt.xticks(rotation=45)

plt.tight_layout()
plt.grid(False)
plt.show()

# Group by vehicle age and gender, adding annual premiums
grouped_data = train_df.groupby(['Vehicle_Age', 'Gender'])['Annual_Premium'].sum().reset_index()

# Grouped bar chart
plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_data, x='Vehicle_Age', y='Annual_Premium', hue='Gender', palette='viridis')
plt.title('Total Annual Premium by Vehicle Age and Sex')
plt.xlabel('Vehicle Age')
plt.ylabel('Annual Premium Total')
plt.legend(title='Genre')
plt.grid(False)
plt.show()

# List of genres
genders = train_df['Gender'].unique()

# Figure size
plt.figure(figsize=(15, 8))

# Loop over genres
for i, gender in enumerate(genders, 1):
 plt.subplot(1, 2, i)
 gender_data = train_df[train_df['Gender'] == gender]
 gender_grouped = gender_data.groupby('Vehicle_Age')['Annual_Premium'].sum().reset_index()
 sns.barplot(data=gender_grouped, x='Vehicle_Age', y='Annual_Premium', palette='viridis')
 plt.title(f'Total Annual Premium by Vehicle Age ({gender})')
 plt.xlabel('Vehicle Age')
 plt.ylabel('Annual Premium Total')

plt.tight_layout()
plt.grid(False)
plt.show()

# Group by vehicle age and insurance status, adding annual premiums
grouped_data = train_df.groupby(['Vehicle_Age', 'Previously_Insured'])['Annual_Premium'].sum().reset_index()

# Convert the 'Previously_Insured' column to string for better visualization
grouped_data['Previously_Insured'] = grouped_data['Previously_Insured'].astype(str)

# Grouped bar chart
plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_data, x='Vehicle_Age', y='Annual_Premium', hue='Previously_Insured', palette='viridis')
plt.title('Total Annual Premium by Vehicle Age and Insurance Status')
plt.xlabel('Vehicle Age')
plt.ylabel('Annual Premium Total')
plt.legend(title='Previously Insured')
plt.grid(False)
plt.show()

# List of insurance status
statuses = train_df['Previously_Insured'].unique()

# Figure size
plt.figure(figsize=(15, 8))

# Group by vehicle age and insurance status, adding annual premiums
grouped_data = train_df.groupby(['Vehicle_Age', 'Previously_Insured'])['Annual_Premium'].sum().reset_index()

# Convert the 'Previously_Insured' column to string for better visualization
grouped_data['Previously_Insured'] = grouped_data['Previously_Insured'].astype(str)

# Grouped bar chart
plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_data, x='Vehicle_Age', y='Annual_Premium', hue='Previously_Insured', palette='viridis')
plt.title('Total Annual Premium by Vehicle Age and Insurance Status')
plt.xlabel('Vehicle Age')
plt.ylabel('Annual Premium Total')
plt.legend(title='Previously Insured')
plt.grid(False)
plt.show()

# List of insurance status
statuses = train_df['Previously_Insured'].unique()

# Figure size
plt.figure(figsize=(15, 8))

# Loop about insurance statuses
for i, status in enumerate(statuses, 1):
 plt.subplot(1, 2, i)
 status_data = df[df['Previously_Insured'] == status]
 status_grouped = status_data.groupby('Vehicle_Age')['Annual_Premium'].sum().reset_index()
 sns.barplot(data=status_grouped, x='Vehicle_Age', y='Annual_Premium', palette='viridis')
 plt.title(f'Total Annual Premium by Vehicle Age (Previously Insured: {status})')
 plt.xlabel('Vehicle Age')
 plt.ylabel('Annual Premium Total')

plt.tight_layout()
plt.grid(False)
plt.show()

# Group by age, gender and vehicle age, adding annual premiums
grouped_data = train_df.groupby(['Age', 'Gender', 'Vehicle_Age'])['Annual_Premium'].sum().reset_index()

# Convert the 'Gender' column to string for better visualization
grouped_data['Gender'] = grouped_data['Gender'].map({'Male': 'Man', 'Female': 'Woman'})

# Configure the size of the figure
plt.figure(figsize=(30.5, 10))

# Loop to create graphs separated by vehicle age
vehicle_ages = grouped_data['Vehicle_Age'].unique()

for i, vehicle_age in enumerate(vehicle_ages, 1):
    plt.subplot(2, 2, i)
    subset = grouped_data[grouped_data['Vehicle_Age'] == vehicle_age]
    sns.barplot(data=subset, x='Age', y='Annual_Premium', hue='Gender', palette='viridis')
    plt.title(f'Total Annual Premium by Age and Sex (Vehicle Age: {vehicle_age})')
    plt.xlabel('Age')
    plt.ylabel('Annual Premium Total')
    plt.legend(title='Genre')
    plt.xticks(rotation=50)

plt.tight_layout()
plt.grid(False)
plt.show()

# Group by vehicle age and gender, adding annual premiums
grouped_data = train_df.groupby(['Vehicle_Age', 'Gender'])['Annual_Premium'].sum().reset_index()

# Grouped bar chart
plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_data, x='Vehicle_Age', y='Annual_Premium', hue='Gender', palette='viridis')
plt.title('Total Annual Premium by Vehicle Age and Sex')
plt.xlabel('Vehicle Age')
plt.ylabel('Annual Premium Total')
plt.legend(title='Genre')
plt.grid(False)
plt.show()

# List of genres
genders = train_df['Gender'].unique()

# Figure size
plt.figure(figsize=(15, 8))

#Loop over genres
for i, gender in enumerate(genders, 1):
 plt.subplot(1, 2, i)
 gender_data = train_df[train_df['Gender'] == gender]
 gender_grouped = gender_data.groupby('Vehicle_Age')['Annual_Premium'].sum().reset_index()
 sns.barplot(data=gender_grouped, x='Vehicle_Age', y='Annual_Premium', palette='viridis')
 plt.title(f'Total Annual Premium by Vehicle Age ({gender})')
 plt.xlabel('Vehicle Age')
 plt.ylabel('Annual Premium Total')

plt.tight_layout()
plt.grid(False)
plt.show()

# Group by vehicle age and insurance status, adding annual premiums
grouped_data = train_df.groupby(['Vehicle_Age', 'Previously_Insured'])['Annual_Premium'].sum().reset_index()

# Convert the 'Previously_Insured' column to string for better visualization
grouped_data['Previously_Insured'] = grouped_data['Previously_Insured'].astype(str)

# Grouped bar chart
plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_data, x='Vehicle_Age', y='Annual_Premium', hue='Previously_Insured', palette='viridis')
plt.title('Total Annual Premium by Vehicle Age and Insurance Status')
plt.xlabel('Vehicle Age')
plt.ylabel('Annual Premium Total')
plt.legend(title='Previously Insured')
plt.grid(False)
plt.show()

# List of insurance status
statuses = train_df['Previously_Insured'].unique()

# Figure size
plt.figure(figsize=(15, 8))

# Loop about insurance statuses
for i, status in enumerate(statuses, 1):
 plt.subplot(1, 2, i)
 status_data = df[df['Previously_Insured'] == status]
 status_grouped = status_data.groupby('Vehicle_Age')['Annual_Premium'].sum().reset_index()
 sns.barplot(data=status_grouped, x='Vehicle_Age', y='Annual_Premium', palette='viridis')
 plt.title(f'Total Annual Premium by Vehicle Age (Previously Insured: {status})')
 plt.xlabel('Vehicle Age')
 plt.ylabel('Annual Premium Total')

plt.tight_layout()
plt.grid(False)
plt.show()

# Group by age, gender and vehicle age, adding annual premiums
grouped_data = train_df.groupby(['Age', 'Gender', 'Vehicle_Age'])['Annual_Premium'].sum().reset_index()

# Convert the 'Gender' column to string for better visualization
grouped_data['Gender'] = grouped_data['Gender'].map({'Male': 'Man', 'Female': 'Woman'})

# Configure the size of the figure
plt.figure(figsize=(30.5, 10))

# Loop to create graphs separated by vehicle age
vehicle_ages = grouped_data['Vehicle_Age'].unique()

for i, vehicle_age in enumerate(vehicle_ages, 1):
    plt.subplot(2, 2, i)
    subset = grouped_data[grouped_data['Vehicle_Age'] == vehicle_age]
    sns.barplot(data=subset, x='Age', y='Annual_Premium', hue='Gender', palette='viridis')
    plt.title(f'Total Annual Premium by Age and Sex (Vehicle Age: {vehicle_age})')
    plt.xlabel('Age')
    plt.ylabel('Annual Premium Total')
    plt.legend(title='Genre')
    plt.xticks(rotation=50)

plt.tight_layout()
plt.grid(False)
plt.show()

# Select the specified columns
columns_of_interest = ["id", "Age", "Driving_License", "Region_Code", "Previously_Insured", "Annual_Premium", "Policy_Sales_Channel", "Vintage", "Response"]
df_selected = df[columns_of_interest]

# Calculate the correlation matrix
correlation_matrix = df_selected.corr()

# Configure the size of the figure
plt.figure(figsize=(14, 10))

# Correlation heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Select numeric columns
numeric_columns = ["Age", "Annual_Premium", "Vintage"]

# Configure the size of the figure
plt.figure(figsize=(18, 6))

# Loop to create boxplots for each numeric column
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=train_df, y=column, palette='viridis')
    plt.title(f'Boxplot of {column}')
    plt.ylabel(column)

plt.tight_layout()
plt.grid(False)
plt.show()

# Select numeric columns
numeric_columns = ["Age", "Annual_Premium", "Vintage"]

# Configure the size of the figure
plt.figure(figsize=(18, 6))

# Loop to create fiddle plots for each numeric column, separated by Response
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=train_df, x='Response', y=column, palette='viridis')
    plt.title(f'Violin Chart of {column} by Response')
    plt.xlabel('Response')
    plt.ylabel(column)

plt.tight_layout()
plt.grid(False)
plt.show()

# Descriptive statistics separated by Response
for column in numeric_columns:
    print(f'\nDescriptive Statistics of {column} by Response:')
    print(train_df.groupby('Response')[column].describe())

# Select numeric columns
numeric_columns = ["Age", "Annual_Premium", "Vintage"]

def remove_outliers(train_df, columns):
    for column in columns:
        Q1 = train_df[column].quantile(0.25)
        Q3 = train_df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        train_df = train_df[(train_df[column] >= lower_bound) & (train_df[column] <= upper_bound)]
    return train_df


# Remove outliers from the DataFrame
df_cleaned = remove_outliers(train_df, numeric_columns)

# Check the DataFrame dimension after removing outliers
print(f"Original dimension: {train_df.shape}")
print(f"Dimension after removing outliers: {df_cleaned.shape}")

# View the first records of the cleaned DataFrame
df_cleaned.head()

# Select numeric columns
numeric_columns = ["Age", "Annual_Premium", "Vintage"]

# Configure the size of the figure
plt.figure(figsize=(18, 6))

# Loop to create fiddle plots for each numeric column, separated by Response
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=df_cleaned, x='Response', y=column, hue="Response", palette='viridis')
    plt.title(f'Violin Chart of {column} by Response')
    plt.xlabel('Response')
    plt.ylabel(column)

plt.tight_layout()
plt.grid(False)
plt.show()

# Descriptive statistics separated by Response
for column in numeric_columns:
    print(f'\nDescriptive Statistics of {column} by Response:')
    print(df_cleaned.groupby('Response')[column].describe())


def optimize_memory_usage(df):
    df = df.copy()

    print("Memory usage before optimization:")
    print(df.memory_usage(deep=True))
    print()

    # Convert columns to 'category' type if they are present
    categorical_columns = [
        'Gender', 'Driving_License', 'Region_Code', 'Previously_Insured',
        'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel',
        'Response', 'Age_Bucket'
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Optimize numeric columns
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], downcast='integer')

    if 'Annual_Premium' in df.columns:
        df['Annual_Premium'] = pd.to_numeric(df['Annual_Premium'], downcast='integer')

    if 'Vintage' in df.columns:
        df['Vintage'] = pd.to_numeric(df['Vintage'], downcast='integer')

    print("Memory usage after optimization:")
    print(df.memory_usage(deep=True))
    print()

    print("DataFrame info after optimization:")
    df.info(memory_usage='deep')
    print()

    return df_cleaned

# Optimizing the DataFrames
df_cleaned_optimized_train = optimize_memory_usage(df_cleaned)
test_df_optimized_test = optimize_memory_usage(test_df)
df_optimized = optimize_memory_usage(df)

# Copy dataset
train_df = df_cleaned_optimized_train.copy()
test_df = df_optimized.copy()

# 1. Handling Missing Values
print("Number of missing values ​​per column:")
print(df_optimized.isnull().sum())

# View missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df_optimized.isnull(), cbar=False, cmap="viridis")
plt.title("Viewing Missing Values in the Training Set")
plt.show()

# Importing library
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
label_encoder = LabelEncoder()
df_optimized['Gender'] = label_encoder.fit_transform(df_optimized['Gender'])
df_optimized['Vehicle_Age'] = label_encoder.fit_transform(df_optimized['Vehicle_Age'])
df_optimized['Vehicle_Damage'] = label_encoder.fit_transform(df_optimized['Vehicle_Damage'])

# Viewing
label_encoder

# View the first DataFrame records after encoding
df_optimized

# Fill missing values
df_optimized.fillna(method='ffill', inplace=True)
df_optimized.fillna(method='ffill', inplace=True)

# Resources
X = df_optimized.drop(columns=['Response'])

# Target variable
y = df_optimized['Response']
# Viewing rows and columns x
X.shape

# Viewing rows and columns
y.shape

# Importing libraries
from sklearn.model_selection import train_test_split

# Training and testing division
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Viewing training data
print("Viewing rows and columns given by X train", X_train.shape)

# Viewing test data
print("Viewing rows and columns given y train", y_train.shape)

# Converting categorical columns to dummy variables
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Viewing training data
print("Viewing rows and columns given by X train", X_train.shape)

# Viewing test data
print("Viewing rows and columns given y train", y_train.shape)

# Importing machine learning model library
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Importing library for metrics machine learning models
from sklearn.metrics import accuracy_score

# Models to be evaluated
models = [GaussianNB(), # Naive Bayes Model
          DecisionTreeClassifier(random_state=42), # Decision Tree Model
          #RandomForestClassifier(n_estimators=100, random_state=42), # Random forest model
          LogisticRegression(random_state=50), # Logistic regression model
          AdaBoostClassifier(random_state=45), # Ada Boost Model
          XGBClassifier(), # XGBoost Model Parameter tree_method='gpu_hist' for XGBoost GPU
          LGBMClassifier()] # LightGBM Model Parameter device='gpu' for LightGBM GPU

# Evaluate each model
for i, model in enumerate(models):
    model.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(model)
    print()
    print(f"Model {i+1}: {type(model).__name__}")
    print()
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Testing Accuracy: {test_accuracy}")
    print("------------------")

# Step 6: Evaluate the model
# Step 7: Make predictions on the test set
predictions = model.predict(X_test)

# Train models that support feature importances

# Set Seaborn style
sns.set_palette("Set2")

models_with_feature_importances = [("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42)),
                                 #  (
                                 #  "RandomForestClassifier", RandomForestClassifier(n_estimators=100, random_state=42)),
                                   ("XGBClassifier", XGBClassifier(random_state=42)),
                                   ("LGBMClassifier", LGBMClassifier(random_state=42))]

# Iterate over models
for model_name, model in models_with_feature_importances:

    # Train model
    model.fit(X_train, y_train)

    # Get importance of features
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    else:
        # If the model does not have feature_importances_, continue to the next model
        print(f"{model_name} does not support feature importances.")
        continue

    # Create DataFrame for easier viewing
    feature_importances_df = pd.DataFrame({'Feature': X_train.columns,
                                           'Importance': feature_importances})

    # Sort by importance
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances_df[:10])
    plt.title(f"Top 10 Features - {model_name}")
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.grid(False)
    plt.show()

# plot confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluate each model
for i, model in enumerate(models):

    # Calculate and plot the confusion matrix
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Not Transported", "Transported"],
                yticklabels=["Not Transported", "Transported"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - Model {i + 1}: {type(model).__name__}")
    plt.show()
    print("------------------")

# ROC curve models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, classification_report, confusion_matrix
)


# Evaluate each model
for i, model in enumerate(models):

    # Calculate positive class probabilities
    y_probs = model.predict_proba(X_test)[:, 1]

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)

    # Calculate the area under the ROC curve (AUC)
    auc = roc_auc_score(y_test, y_probs)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Model {i + 1}: {type(model).__name__}')
    plt.legend(loc="lower right")
    plt.grid(False)
    plt.show()

    print("------------------")
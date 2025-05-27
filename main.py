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
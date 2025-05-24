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
# Importing Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("dataSet CA 2 INT375.csv")

# Displaying initial info
print("Initial Dataset Info:")
print(data.info())
print("\nSample Data:")
print(data.head())

# Fill missing values: Construction Date with Mean
mean_year = data['CONSTRUCTION DATE'].mean()
data['CONSTRUCTION DATE'].fillna(mean_year, inplace=True)

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(data.describe(include='all'))

# Set style for all plots
sns.set(style="whitegrid")

# Bar Plot: Building Status
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='BUILDING STATUS')
plt.title("Building Status Distribution")
plt.xlabel("Status")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Pie Chart: Asset Type Distribution
plt.figure(figsize=(8, 6))
asset_counts = data['REAL PROPERTY ASSET TYPE'].value_counts()
plt.pie(asset_counts, labels=asset_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Asset Type Distribution")
plt.axis('equal')
plt.tight_layout()
plt.show()

# Histogram: Construction Date
plt.figure(figsize=(8, 6))
sns.histplot(data['CONSTRUCTION DATE'], kde=True, bins=30, color='skyblue')
plt.title("Distribution of Construction Years")
plt.xlabel("Construction Year")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Boxplot: Rentable Square Feet
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['BUILDING RENATABLE SQUARE FEET'], color='lightgreen')
plt.title("Boxplot of Rentable Square Feet")
plt.xlabel("Rentable Square Feet")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

# Line Plot: Average Rentable Square Feet by Region
region_avg = data.groupby('GSA REGION')['BUILDING RENATABLE SQUARE FEET'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=region_avg, x='GSA REGION', y='BUILDING RENATABLE SQUARE FEET', marker='o')
plt.title("Average Rentable Area by GSA Region")
plt.xlabel("GSA Region")
plt.ylabel("Average Rentable Sq Ft")
plt.tight_layout()
plt.show()

# Regression Plot: Construction Date vs Rentable Sq Ft
plt.figure(figsize=(10, 6))
sns.regplot(x='CONSTRUCTION DATE', y='BUILDING RENATABLE SQUARE FEET', data=data, scatter_kws={'alpha':0.3}, line_kws={'color': 'red'})
plt.title("Construction Year vs Rentable Area")
plt.xlabel("Construction Year")
plt.ylabel("Rentable Square Feet")
plt.tight_layout()
plt.show()

# Linear Regression Model
X = data[['CONSTRUCTION DATE']]
y = data['BUILDING RENATABLE SQUARE FEET']
model = LinearRegression()
model.fit(X, y)

# Print model details
print("\nLinear Regression Model Results:")
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
print("R-squared Score:", model.score(X, y))

# ML-Assignment-1-Statistical-Measures
This  Assignment explains about the Statistical Measures using Python
import warnings
warnings.filterwarnings("ignore")
# Q1. Perform basic EDA 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
df=pd.read_csv("house_price.csv")
df

print("Shape of the dataset:")
print(df.shape)
print("Dataset Info:")
print(df.info())

print("Statistical Summary:")
print(df.describe())

print("Null values in each column:")
print(df.isnull().sum())

# Q2. Detect the outliers using following methods and remove it using methods like trimming / capping/ imputation using mean or median 
a) Mean and Standard deviation 
b) Percentile method 
c) IQR(Inter quartile range method) 
d) Z Score method

df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(df.shape)

# To understand the dataset is symmetric or assymeric 
# Read dataset
df = pd.read_csv("house_price.csv")

# Plot KDE for both columns
plt.figure(figsize=(8, 5))
sns.kdeplot(df["price"], color='blue',  label="price_per_sqft",linewidth=3)


# Labels and Title
plt.xlabel("Value", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.title("KDE Plot for Normally Distributed Data", fontsize=14)
plt.legend()

# Show Plot
plt.show()

# To find the skewness of each columns in the dataset
df[["price_per_sqft","total_sqft","bath","price","bhk"]].skew()

# To reduce the skewness of the dataset,using sqaure root transformation 
# Load data
df = pd.read_csv("house_price.csv")

# Check original skewness
print("Original Skewness:\n", df[["price_per_sqft", "total_sqft", "bath", "price", "bhk"]].skew())

# Apply square root transformation to selected columns
df[['price_per_sqft_sqrt', 'total_sqft_sqrt', 'bath_sqrt', 'price_sqrt', 'bhk_sqrt']] = np.sqrt(df[["price_per_sqft", "total_sqft", "bath", "price", "bhk"]])

# Check skewness after transformation
print("Square Root Skewness:\n", df[['price_per_sqft_sqrt', 'total_sqft_sqrt', 'bath_sqrt', 'price_sqrt', 'bhk_sqrt']].skew())


# Plot the distributions
plt.figure(figsize=(10, 5))

# Original data distribution (Example: price column)
plt.subplot(1, 2, 1)
plt.hist(df['price'], bins=10, color='blue', alpha=0.7)
plt.title("Original Data")

# Transformed data distribution (Example: price_sqrt column)
plt.subplot(1, 2, 2)
plt.hist(df['price_sqrt'], bins=10, color='green', alpha=0.7)
plt.title("Square Root Transformed Data")

plt.show()
# Apply second square root transformation
df[['price_per_sqft_sqrt2', 'total_sqft_sqrt2', 'bath_sqrt2', 'price_sqrt2', 'bhk_sqrt2']] = np.sqrt(df[['price_per_sqft_sqrt', 'total_sqft_sqrt', 'bath_sqrt', 'price_sqrt', 'bhk_sqrt']])

# Check skewness after second transformation
print("Second Square Root Skewness:\n", df[['price_per_sqft_sqrt2', 'total_sqft_sqrt2', 'bath_sqrt2', 'price_sqrt2', 'bhk_sqrt2']].skew())

# Plotting the Box Plot for price_per_sqft
plt.figure(figsize=(10, 6))

# Box plot for 
sns.boxplot(x=newdf['price_per_sqft'],orient='h')

# Adding title and labels
plt.title('Box Plot for price_per_sqft before outlier removal', fontsize=14)
plt.xlabel('"Price per Square Foot ', fontsize=12)
plt.ylabel('Features (price_per_sqft)', fontsize=12)

# Display the plot

plt.xlabel("Price per Square Foot")
plt.show()

# The dataset is not symmetric ,so using IQR method for Detecting outliers
q1 = df['price_per_sqft'].quantile(0.25)
q3 = df['price_per_sqft'].quantile(0.75)
iqr=q3-q1
upper_limit = q3 +(1.5*iqr)
lower_limit =  q1 - (1.5*iqr)
print(upper_limit, lower_limit)
print(q1,q3,iqr)

# Find the outliers
df.loc[(df['price_per_sqft'] > upper_limit) | (df['price_per_sqft'] < lower_limit)]

#trimming - deleting the outliers
newdf= df.loc[(df['price_per_sqft'] < upper_limit) & (df['price_per_sqft'] > lower_limit)]
print("before removing the data:", len(df))
print("after removing the data:", len(newdf))
print("outliers:", len(df)-len(newdf))

# Q3. Create a box plot and use this to determine which method seems to work best to remove outliers for this data?


# Plotting the Box Plot for price_per_sqft'
plt.figure(figsize=(10, 6))

# Box plot for price_per_sqft 
sns.boxplot(x=newdf['price_per_sqft'],orient='h')  

# Adding title and labels
plt.title('Box Plot for price_per_sqft after outlier removal', fontsize=14)
plt.xlabel('Price per Square Foot', fontsize=12)
plt.ylabel('Features (price_per_sqft)', fontsize=12)

# Display the plot
plt.show()

# Q4. Draw histplot to check the normality of the column(price per sqft column) and perform transformations if needed. Check the skewness and kurtosis before and after the transformation. 

# Load dataset
df = pd.read_csv("House_price.csv")

# Check skewness and kurtosis before transformation
skew_before = df["price_per_sqft"].skew()
kurt_before = df["price_per_sqft"].kurtosis()

# Plot histogram before transformation
plt.figure(figsize=(12, 5))
sns.histplot(df["price_per_sqft"], bins=50, kde=True)
plt.title("Histogram of price_per_sqft (Before Transformation)")
plt.xlabel("Price per Sqft")
plt.ylabel("Frequency")
plt.show()

# Apply log transformation
df["price_per_sqft_log"] = np.log1p(df["price_per_sqft"])

# Check skewness and kurtosis after transformation
skew_after = df["price_per_sqft_log"].skew()
kurt_after = df["price_per_sqft_log"].kurtosis()

# Plot histogram after transformation
plt.figure(figsize=(12, 5))
sns.histplot(df["price_per_sqft_log"], bins=50, kde=True)
plt.title("Histogram of Log-Transformed price_per_sqft")
plt.xlabel("Log(Price per Sqft)")
plt.ylabel("Frequency")
plt.show()

# Display skewness and kurtosis before and after transformation
print(f"Skewness Before: {skew_before}, Kurtosis Before: {kurt_before}")
print(f"Skewness After: {skew_after}, Kurtosis After: {kurt_after}")



# Q5. Check the correlation between all the numerical columns and plot heatmap.

# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("house_price.csv")

# Compute correlation matrix for numerical columns
correlation_matrix = df.corr(numeric_only=True)  # Ensure only numerical columns are considered

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()


# Q6. Draw Scatter plot between the variables to check the correlation between them. 

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.scatterplot(x=df['total_sqft'], y=df['price'], ax=axes[0, 0])
axes[0, 0].set_title("Total Sqft vs Price")

sns.scatterplot(x=df['bhk'], y=df['price'], ax=axes[0, 1])
axes[0, 1].set_title("BHK vs Price")

sns.scatterplot(x=df['bath'], y=df['price'], ax=axes[1, 0])
axes[1, 0].set_title("Bathrooms vs Price")

sns.scatterplot(x=df['price_per_sqft'], y=df['total_sqft'], ax=axes[1, 1])
axes[1, 1].set_title("Price per Sqft vs Total Sqft")

plt.tight_layout()
plt.show()

--The End--


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# LAB 1 : Dataset Exploration

print("=" * 50)
print("LOAD DATASET")
print("=" * 50)

df = pd.read_csv("WineQT.csv")

print(df.head())

# Shape
print("\nSHAPE")
print(df.shape)

# Data Types
print("\nDATA TYPES")
print(df.dtypes)

# Summary Statistics
print("\nSUMMARY STATISTICS")
print(df.describe())

# Missing Values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Duplicate Records
print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# Class Distribution
print("\nCLASS DISTRIBUTION")
print(df["quality"].value_counts())

# LAB 2 : Data Visualization

# Histogram
plt.figure(figsize=(15,10))
df.hist(figsize=(15,10))
plt.suptitle("Wine Dataset Histogram")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# PART 3 : Data Cleaning

print("\n")
print("=" * 50)
print("DATA CLEANING")
print("=" * 50)

# Missing Value Handling
print("\nMissing Before:")
print(df.isnull().sum())

df = df.fillna(df.mean(numeric_only=True))

print("\nMissing After:")
print(df.isnull().sum())

# Duplicate Removal
print("\nRows Before Duplicate Removal:")
print(df.shape)

df = df.drop_duplicates()

print("\nRows After Duplicate Removal:")
print(df.shape)

# Incorrect Data Correction

numeric_cols = df.select_dtypes(include=["number"]).columns

for col in numeric_cols:
    if col != "Id":
        df = df[df[col] >= 0]

print("\nIncorrect Data Corrected")

# Data Type Conversion

df["quality"] = df["quality"].astype(int)

print("\nData Types After Conversion")
print(df.dtypes)

# Mean
print("\nMEAN")
print(df.mean(numeric_only=True))

# Median
print("\nMEDIAN")
print(df.median(numeric_only=True))

# PART 4 

print("\n")
print("=" * 50)
print("FEATURE ENGINEERING")
print("=" * 50)

# Label Encoding

label_encoder = LabelEncoder()

df["quality_encoded"] = label_encoder.fit_transform(
    df["quality"]
)

print("\nLABEL ENCODING")
print(
    df[
        ["quality",
         "quality_encoded"]
    ].head()
)

# One Hot Encoding

df_onehot = pd.get_dummies(
    df,
    columns=["quality"],
    prefix="quality"
)

print("\nONE HOT ENCODING")
print(df_onehot.head())

# FEATURE SCALING

print("\n")
print("=" * 50)
print("STANDARD SCALER")
print("=" * 50)

scaler = StandardScaler()

feature_columns = [
    col for col in df.columns
    if col not in [
        "quality_encoded",
        "Id"
    ]
]

scaled_data = scaler.fit_transform(
    df[feature_columns]
)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=feature_columns
)

print(scaled_df.head())

# SAVE CLEAN DATA

df.to_csv(
    "WineQT_Cleaned.csv",
    index=False
)

print("\nSaved : WineQT_Cleaned.csv")

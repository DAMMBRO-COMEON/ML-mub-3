# ==========================================
# ML LAB 02 : Data Preprocessing
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ==========================================
# CREATE DATASET
# ==========================================

data = {
    "ID": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    "Name": ["John","Sarah","Michael","Emma","David",
             "Olivia","James","Sophia","Daniel","Ava",
             "John","William","Mia","Ethan","Charlotte"],
    "Gender": ["M","F","M","F","M",
               "F","M","F","M","F",
               "M","male","female","M","F"],
    "Age": [25,28,35,22,np.nan,
            27,31,24,29,26,
            25,33,21,40,23],
    "Salary": [25000,32000,45000,22000,38000,
               np.nan,41000,27000,35000,30000,
               25000,47000,21000,55000,24000],
    "Department": ["IT","HR","Finance","IT","Marketing",
                   "HR","Finance","Marketing","IT","HR",
                   "IT","Finance","Marketing","IT","HR"],
    "Status": ["Pass","Pass","Pass","Fail","Pass",
               "Pass","Pass","Fail","Pass","Pass",
               "Pass","Pass","Fail","Pass","Fail"]
}

df = pd.DataFrame(data)

# บันทึก dataset
df.to_csv("dataset.csv", index=False)

print("dataset.csv created successfully")

# ==========================================
# PART 1 : DATASET EXPLORATION
# ==========================================

print("\n" + "="*50)
print("DATASET EXPLORATION")
print("="*50)

print("\n1. Shape")
print(df.shape)

print("\n2. Data Types")
print(df.dtypes)

print("\n3. Summary Statistics")
print(df.describe(include="all"))

print("\n4. Missing Values")
print(df.isnull().sum())

print("\n5. Duplicate Records")
print(df.duplicated().sum())

print("\n6. Class Distribution")
print(df["Status"].value_counts())

# ==========================================
# PART 2 : DATA VISUALIZATION
# ==========================================

df.select_dtypes(include=np.number).hist(
    figsize=(10,8),
    bins=10
)

plt.suptitle("Histogram")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# ==========================================
# PART 3 : DATA CLEANING
# ==========================================

print("\n" + "="*50)
print("DATA CLEANING")
print("="*50)

print("\nMissing Values Before")
print(df.isnull().sum())

# Handling Missing Values
for col in df.select_dtypes(include=np.number):
    df[col] = df[col].fillna(df[col].mean())

for col in df.select_dtypes(include="object"):
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After")
print(df.isnull().sum())

# Remove Duplicate
before = len(df)

df.drop_duplicates(inplace=True)

after = len(df)

print("\nDuplicates Removed =", before - after)

# Correct Gender Values
df["Gender"] = df["Gender"].replace({
    "M": "Male",
    "F": "Female",
    "male": "Male",
    "female": "Female",
    "m": "Male",
    "f": "Female"
})

print("\nGender values corrected")

# ==========================================
# Mean vs Median
# ==========================================

print("\n" + "="*50)
print("MEAN VS MEDIAN")
print("="*50)

for col in df.select_dtypes(include=np.number):

    print(f"\nColumn: {col}")
    print("Mean   =", round(df[col].mean(),2))
    print("Median =", round(df[col].median(),2))

# ==========================================
# PART 4 : FEATURE ENGINEERING
# ==========================================

print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Label Encoding
label_df = df.copy()

le = LabelEncoder()

for col in label_df.select_dtypes(include="object").columns:
    label_df[col] = le.fit_transform(label_df[col])

print("\nLabel Encoding Result")
print(label_df.head())

# One Hot Encoding
onehot_df = pd.get_dummies(
    df,
    columns=["Gender","Department","Status"]
)

print("\nOne Hot Encoding Result")
print(onehot_df.head())

# Save Clean Data
df.to_csv("cleaned_dataset.csv", index=False)

print("\ncleaned_dataset.csv saved")
print("\nLab Completed Successfully")
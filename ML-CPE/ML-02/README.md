# ML LAB 02 - Data Preprocessing

Machine Learning Lab Assignment focusing on Data Preprocessing techniques using Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn.

---

## Project Overview

This project demonstrates the fundamental steps of data preprocessing before applying Machine Learning algorithms.

The workflow includes:

- Dataset Exploration
- Data Visualization
- Data Cleaning
- Feature Engineering

The objective is to understand how to inspect, clean, transform, and prepare data for machine learning models.

---

## Dataset Information

| Column | Description |
|----------|------------|
| ID | Employee ID |
| Name | Employee Name |
| Gender | Gender Information |
| Age | Employee Age |
| Salary | Employee Salary |
| Department | Department Name |
| Status | Pass / Fail Class Label |

Dataset Size:

- Rows: 15
- Columns: 7

---

# Part 1: Dataset Exploration

### Dataset Shape

```
(15, 7)
```

### Missing Values

```
Age       1
Salary    1
```

### Duplicate Records

```
0
```

### Class Distribution

```
Pass = 11
Fail = 4
```

---

# Part 2: Data Visualization

## Histogram

The histogram was generated to inspect the distribution of numerical features:

- ID
- Age
- Salary

## Correlation Heatmap

Correlation matrix:

| Feature | Correlation |
|----------|------------|
| Age ↔ Salary | 0.98 |
| ID ↔ Age | 0.06 |
| ID ↔ Salary | 0.09 |

Observation:

- Age and Salary have a very strong positive correlation.
- Salary tends to increase as Age increases.

---

# Part 3: Data Cleaning

## Missing Value Handling

Missing values in Age and Salary were replaced using Mean Imputation.

### Before Cleaning

```
Age       1
Salary    1
```

### After Cleaning

```
Age       0
Salary    0
```

## Duplicate Removal

```
Duplicates Removed = 0
```

## Incorrect Data Correction

Gender values were standardized:

```
M, male -> Male
F, female -> Female
```

## Mean vs Median

### Age

```
Mean   = 27.79
Median = 27.00
```

### Salary

```
Mean   = 33357.14
Median = 32000.00
```

---

# Part 4: Feature Engineering

## Label Encoding

Categorical variables were converted into numerical values.

## One-Hot Encoding

Generated features include:

```
Gender_Female
Gender_Male
Department_Finance
Department_HR
Department_IT
Department_Marketing
Status_Fail
Status_Pass
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

# How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python Python.py
```

---

# Results Summary

✅ Dataset Exploration completed

✅ Missing Values handled

✅ Duplicate Checking completed

✅ Data Standardization completed

✅ Mean vs Median analysis completed

✅ Label Encoding completed

✅ One-Hot Encoding completed

✅ Cleaned dataset exported successfully

---

# อ้างอิง (References)

1. Pandas Documentation — https://pandas.pydata.org/docs/
2. NumPy Documentation — https://numpy.org/doc/
3. Scikit-Learn Documentation — https://scikit-learn.org/stable/
4. Matplotlib Documentation — https://matplotlib.org/stable/
5. Seaborn Documentation — https://seaborn.pydata.org/
6. Han, Kamber & Pei. Data Mining: Concepts and Techniques (3rd Edition).

---

# สรุปภาษาไทย

โครงงานนี้เป็นการศึกษา Data Preprocessing สำหรับงาน Machine Learning โดยประกอบด้วยการสำรวจข้อมูล การสร้างกราฟ การจัดการ Missing Values การตรวจสอบข้อมูลซ้ำ การแก้ไขข้อมูลที่ไม่สอดคล้องกัน และการทำ Feature Engineering ด้วย Label Encoding และ One-Hot Encoding เพื่อเตรียมข้อมูลให้พร้อมสำหรับการสร้างโมเดล Machine Learning.

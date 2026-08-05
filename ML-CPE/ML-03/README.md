# Wine Quality Analysis Using Regression and Classification

## Author

**วีรภัทร หวานดี**

---

# English Version

## Project Overview

This project focuses on Wine Quality Analysis using Machine Learning techniques. The objectives are:

1. Predict wine quality scores using Regression models.
2. Classify wines as Good Wine or Bad Wine using Classification models.
3. Compare model performance and analyze important wine attributes.

The WineQT dataset was used for this study.

---

## Dataset Information

Dataset: WineQT

Total Instances: **1,143**

Total Features: **11 Input Features**

Target Variable:

- Regression: `quality`
- Classification: `quality_label`

### Features

- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol

---

# Lab 1: Regression Analysis

## Objective

Predict wine quality scores using Linear Regression models.

### 1. Simple Linear Regression

Feature Used: `alcohol`

### Results

- Test MSE: 0.4175
- Test R²: 0.2497

### Interpretation

The model explains approximately 24.97% of the variance in wine quality using only alcohol content.

---

## 2. Multiple Linear Regression

### Results

- Test MSE: 0.3800
- Test R²: 0.3171

### Interpretation

The Multiple Linear Regression model performs better than Simple Linear Regression because it utilizes all available features.

---

# Feature Importance Analysis

Top influential features:

- Density (-30.653)
- Chlorides (-1.807)
- Volatile Acidity (-1.336)
- Sulphates (0.973)
- Citric Acid (-0.339)
- Alcohol (0.267)

---

# Lab 2: Classification Analysis

## Objective

Classify wines into:

- Good Wine (quality >= 6)
- Bad Wine (quality < 6)

Model: Logistic Regression

## Classification Performance

- Accuracy: 77.73%
- Precision: 80.17%
- Recall: 78.23%
- F1-Score: 79.18%

## Confusion Matrix

- TN = 81
- FP = 24
- FN = 27
- TP = 97

---

# Lab 3: Model Comparison

## Regression Comparison

- Simple Linear Regression: MSE = 0.4175, R² = 0.2497
- Multiple Linear Regression: MSE = 0.3800, R² = 0.3171

## Overfitting Check

- Training: MSE = 0.4151, R² = 0.3822
- Testing: MSE = 0.3800, R² = 0.3171

---

# Final Conclusion

- Multiple Linear Regression outperformed Simple Linear Regression.
- Logistic Regression achieved 77.73% classification accuracy.
- Density, Chlorides, Volatile Acidity, Sulphates, and Alcohol were the most influential features.

---

# ภาษาไทย

## ภาพรวมโครงการ

โครงการนี้เป็นการวิเคราะห์คุณภาพไวน์โดยใช้เทคนิค Machine Learning ทั้งแบบ Regression และ Classification

วัตถุประสงค์หลักคือ

1. ทำนายคะแนนคุณภาพไวน์
2. จำแนกไวน์เป็น Good Wine และ Bad Wine
3. เปรียบเทียบประสิทธิภาพของโมเดล

---

## ข้อมูลชุดข้อมูล

- ชุดข้อมูล: WineQT
- จำนวนข้อมูล: 1,143 ตัวอย่าง
- จำนวนตัวแปร: 11 Features

ตัวแปรเป้าหมาย:

- quality (Regression)
- quality_label (Classification)

---

# Lab 1: Regression

## ผลลัพธ์

### Simple Linear Regression

- MSE = 0.4175
- R² = 0.2497

### Multiple Linear Regression

- MSE = 0.3800
- R² = 0.3171

สรุป: Multiple Linear Regression มีประสิทธิภาพดีกว่า

---

# การวิเคราะห์ปัจจัยสำคัญ

ปัจจัยที่มีผลต่อคุณภาพไวน์มากที่สุด ได้แก่

1. Density
2. Chlorides
3. Volatile Acidity
4. Sulphates
5. Alcohol

---

# Lab 2: Classification

## ผลลัพธ์

- Accuracy = 77.73%
- Precision = 80.17%
- Recall = 78.23%
- F1-Score = 79.18%

### Confusion Matrix

- ทำนาย Bad Wine ถูกต้อง = 81
- ทำนาย Good Wine ถูกต้อง = 97
- ทำนายผิดเป็น Good Wine = 24
- ทำนายผิดเป็น Bad Wine = 27

---

# Lab 3: เปรียบเทียบโมเดล

Multiple Linear Regression ให้ผลลัพธ์ดีกว่า Simple Linear Regression และไม่พบ Overfitting ที่รุนแรง

---

# สรุปผลการทดลอง

- Multiple Linear Regression เหมาะสำหรับการทำนายคะแนนคุณภาพไวน์
- Logistic Regression เหมาะสำหรับการจำแนก Good Wine และ Bad Wine
- ความแม่นยำของการจำแนกอยู่ที่ 77.73%
- ปัจจัยสำคัญที่สุด ได้แก่ Density, Chlorides, Volatile Acidity, Sulphates และ Alcohol

---

## Libraries Used

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Run Project

```bash
python wine_quality_analysis.py
```

# ML LAB 05 - Support Vector Machine (SVM)

Machine Learning Lab Assignment focusing on Support Vector Machine (SVM) classification using the WineQT dataset and comparing different kernel functions using Scikit-Learn.

## Project Overview

This project demonstrates the implementation of Support Vector Machine (SVM) for multiclass classification.

The workflow includes:

- Dataset Exploration
- Data Preprocessing
- Feature Standardization
- Model Training
- Model Evaluation
- Kernel Comparison

The objective is to classify wine quality using different SVM kernels and determine which kernel provides the best classification performance.

## Dataset Information

### Dataset Shape

- Rows: 1143
- Columns: 13

### Target Variable

- Quality

### Removed Feature

- Id

## Part 1: Dataset Exploration

### First 5 Rows

The dataset contains physicochemical attributes of red wine samples and their quality scores.

### Dataset Characteristics

- Numerical dataset
- Multi-class classification problem
- Wine quality prediction and classification

## Part 2: Data Preprocessing

### Feature Selection

Input Features:

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

Target Feature:

- Quality

### Train-Test Split

```text
Training Data = 80%
Testing Data  = 20%
```

### Feature Standardization

StandardScaler was applied before training the SVM models because SVM is sensitive to feature scales.

## Part 3: Model Training

### SVM Kernels Evaluated

1. Linear Kernel
2. Polynomial Kernel
3. RBF (Radial Basis Function) Kernel

## Part 4: Model Evaluation

### Accuracy Results

| Kernel | Accuracy |
|----------|----------|
| Linear | 0.6114 |
| Polynomial | 0.6376 |
| RBF | 0.6681 |

### Best Model

```text
Kernel   : RBF
Accuracy : 0.6681
```

## Part 5: Sample Predictions

| Actual | Predicted |
|----------|----------|
| 6 | 7 |
| 5 | 5 |
| 5 | 5 |
| 5 | 5 |
| 5 | 6 |
| 6 | 6 |
| 6 | 5 |
| 6 | 6 |
| 6 | 6 |
| 4 | 5 |

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn


## Results Summary

✅ Dataset loaded successfully

✅ Feature selection completed

✅ Dataset standardized successfully

✅ SVM Linear Kernel trained

✅ SVM Polynomial Kernel trained

✅ SVM RBF Kernel trained

✅ Accuracy comparison completed

✅ Best kernel identified successfully

## Conclusion

The WineQT dataset was successfully classified using Support Vector Machine (SVM).

Experimental results show:

- Linear Kernel Accuracy = 61.14%
- Polynomial Kernel Accuracy = 63.76%
- RBF Kernel Accuracy = 66.81%

Among the evaluated models, the RBF Kernel achieved the highest classification performance and is therefore the most suitable model for the WineQT dataset.

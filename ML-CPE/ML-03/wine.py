import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# LOAD DATASET

df = pd.read_csv("WineQT.csv")

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print(df.info())
print(df.head())

# LAB 1 : REGRESSION

print("\n" + "=" * 50)
print("LAB 1 : REGRESSION (WINE QUALITY PREDICTION)")
print("=" * 50)

target_reg = "quality"

X_reg = df.drop(columns=["quality", "Id"])
y_reg = df["quality"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

# 1.1 Simple Linear Regression

feature_simple = "alcohol"

X_train_simple = X_train_r[[feature_simple]]
X_test_simple = X_test_r[[feature_simple]]

simple_model = LinearRegression()

simple_model.fit(X_train_simple, y_train_r)

y_pred_train_simple = simple_model.predict(X_train_simple)
y_pred_test_simple = simple_model.predict(X_test_simple)

mse_simple = mean_squared_error(y_test_r, y_pred_test_simple)
r2_simple = r2_score(y_test_r, y_pred_test_simple)

print("\nSimple Linear Regression")
print(f"Feature Used : {feature_simple}")
print(f"Test MSE     : {mse_simple:.4f}")
print(f"Test R²      : {r2_simple:.4f}")

# 1.2 Multiple Linear Regression

multi_model = LinearRegression()

multi_model.fit(X_train_r, y_train_r)

y_pred_train_multi = multi_model.predict(X_train_r)
y_pred_test_multi = multi_model.predict(X_test_r)

mse_multi = mean_squared_error(y_test_r, y_pred_test_multi)
r2_multi = r2_score(y_test_r, y_pred_test_multi)

print("\nMultiple Linear Regression")
print(f"Test MSE : {mse_multi:.4f}")
print(f"Test R²  : {r2_multi:.4f}")

# LAB 2 : CLASSIFICATION

print("\n" + "=" * 50)
print("LAB 2 : CLASSIFICATION (GOOD/BAD WINE)")
print("=" * 50)

df["quality_label"] = (df["quality"] >= 6).astype(int)

X_cls = df.drop(columns=["quality", "quality_label", "Id"])
y_cls = df["quality_label"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls,
    y_cls,
    test_size=0.2,
    random_state=42,
    stratify=y_cls
)

# Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_c)
X_test_scaled = scaler.transform(X_test_c)

# Logistic Regression

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_scaled, y_train_c)

y_pred_cls = log_model.predict(X_test_scaled)

# Metrics

acc = accuracy_score(y_test_c, y_pred_cls)
prec = precision_score(y_test_c, y_pred_cls)
rec = recall_score(y_test_c, y_pred_cls)
f1 = f1_score(y_test_c, y_pred_cls)

print("\nLogistic Regression Performance")

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")

# Confusion Matrix

cm = confusion_matrix(y_test_c, y_pred_cls)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Bad Wine", "Good Wine"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix - Wine Classification")
plt.show()

# Decision Boundary Visualization

print("\nGenerating Decision Boundary...")

feat1_idx = 0
feat2_idx = 1

X_2d = X_train_scaled[:, [feat1_idx, feat2_idx]]
y_2d = y_train_c.values

clf_2d = LogisticRegression(max_iter=1000)

clf_2d.fit(X_2d, y_2d)

x_min = X_2d[:, 0].min() - 1
x_max = X_2d[:, 0].max() + 1

y_min = X_2d[:, 1].min() - 1
y_max = X_2d[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3,
    cmap="coolwarm"
)

plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=y_2d,
    cmap="coolwarm",
    edgecolors="k"
)

plt.xlabel(X_cls.columns[feat1_idx])
plt.ylabel(X_cls.columns[feat2_idx])

plt.title("Decision Boundary (Logistic Regression)")

plt.show()

# LAB 3 : MODEL COMPARISON

print("\n" + "=" * 50)
print("LAB 3 : MODEL COMPARISON")
print("=" * 50)

# Regression Comparison

reg_comparison = pd.DataFrame({
    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression"
    ],
    "Test MSE": [
        mse_simple,
        mse_multi
    ],
    "Test R²": [
        r2_simple,
        r2_multi
    ]
})

print("\n1. Regression Comparison")
print(reg_comparison)

# Overfitting Check

train_vs_test = pd.DataFrame({
    "Dataset": [
        "Training",
        "Testing"
    ],
    "MSE": [
        mean_squared_error(
            y_train_r,
            y_pred_train_multi
        ),
        mse_multi
    ],
    "R² Score": [
        r2_score(
            y_train_r,
            y_pred_train_multi
        ),
        r2_multi
    ]
})

print("\n2. Training vs Testing")
print(train_vs_test)

# Summary Table

summary_metrics = pd.DataFrame({
    "Task Type": [
        "Regression",
        "Classification"
    ],
    "Primary Model": [
        "Multiple Linear Regression",
        "Logistic Regression"
    ],
    "Target": [
        "Wine Quality Score",
        "Good/Bad Wine"
    ],
    "Metrics": [
        "MSE, R²",
        "Accuracy, Precision, Recall, F1"
    ]
})

print("\n3. Summary")
print(summary_metrics)

# Feature Importance (Regression)

coef_df = pd.DataFrame({
    "Feature": X_reg.columns,
    "Coefficient": multi_model.coef_
})

coef_df["Abs_Coefficient"] = abs(coef_df["Coefficient"])

coef_df = coef_df.sort_values(
    by="Abs_Coefficient",
    ascending=False
)

print("\nTop Features Affecting Quality")
print(coef_df.head(10))

plt.figure(figsize=(10, 6))

sns.barplot(
    x="Coefficient",
    y="Feature",
    data=coef_df.head(10)
)

plt.title("Top 10 Features Affecting Wine Quality")

plt.tight_layout()

plt.show()
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

# ==========================================
# 0. LOAD DATASET
# ==========================================
# เปลี่ยนชื่อไฟล์ให้ตรงกับไฟล์ CSV ของคุณ
# ตัวอย่างนี้สมมติว่าเป็นชุดข้อมูลที่มีข้อมูลกายภาพ/ลักษณะบุคคล
df = pd.read_csv(r'C:\Users\SW02\Desktop\ML-mub-3\ML-CPE\ML-03\winequality.csv')

print("--- DATASET OVERVIEW ---")
print(df.info())
print(df.head())


# ==========================================
# LAB 1: REGRESSION (Age Prediction)
# ==========================================
print("\n" + "="*50)
print("LAB 1: REGRESSION (AGE PREDICTION)")
print("="*50)

# กำหนด Target สำหรับ Regression คือ 'Age'
# สมมติ feature ที่ดีที่สุดตัวเดียวคือ 'Height' หรือฟีเจอร์อื่นๆ ในไฟล์คุณ
target_reg = 'Age'
X_reg_all = df.drop(columns=[target_reg, 'Gender'], errors='ignore')
y_reg = df[target_reg]

# Split Data (Train / Test)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg_all, y_reg, test_size=0.2, random_state=42
)

# 1.1 Simple Linear Regression (ใช้ 1 Feature เช่น 'Height')
feature_simple = 'Height'  # <--- เปลี่ยนเป็นชื่อ feature ตัวเดียวในไฟล์คุณ
X_train_simple = X_train_r[[feature_simple]]
X_test_simple = X_test_r[[feature_simple]]

model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train_r)

# Prediction & Metrics
y_pred_train_sim = model_simple.predict(X_train_simple)
y_pred_test_sim = model_simple.predict(X_test_simple)

mse_sim_test = mean_squared_error(y_test_r, y_pred_test_sim)
r2_sim_test = r2_score(y_test_r, y_pred_test_sim)

print(f"[Simple Linear Regression - Feature: {feature_simple}]")
print(f"Test MSE: {mse_sim_test:.4f}")
print(f"Test R² : {r2_sim_test:.4f}")

# 1.2 Multiple Linear Regression (ใช้ทุก Features)
model_multi = LinearRegression()
model_multi.fit(X_train_r, y_train_r)

y_pred_train_multi = model_multi.predict(X_train_r)
y_pred_test_multi = model_multi.predict(X_test_r)

mse_multi_test = mean_squared_error(y_test_r, y_pred_test_multi)
r2_multi_test = r2_score(y_test_r, y_pred_test_multi)

print(f"\n[Multiple Linear Regression - All Features]")
print(f"Test MSE: {mse_multi_test:.4f}")
print(f"Test R² : {r2_multi_test:.4f}")


# ==========================================
# LAB 2: CLASSIFICATION (Gender Prediction)
# ==========================================
print("\n" + "="*50)
print("LAB 2: CLASSIFICATION (GENDER PREDICTION)")
print("="*50)

# 2.1 Preparing Classification Data
# แปลง Gender เป็น 0/1 (เช่น Male=1, Female=0 หรือแปลงคำจาก string)
if df['Gender'].dtype == 'object':
    df['Gender_label'] = df['Gender'].map({'Male': 1, 'Female': 0})
else:
    df['Gender_label'] = df['Gender']

target_cls = 'Gender_label'
X_cls = df.drop(columns=['Gender', 'Gender_label'], errors='ignore')
y_cls = df[target_cls]

# Split Data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

# Feature Scaling (จำเป็นสำหรับ Logistic Regression)
scaler = StandardScaler()
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)

# 2.2 Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train_c_scaled, y_train_c)

y_pred_c = log_reg.predict(X_test_c_scaled)

# 2.3 Confusion Matrix & Performance Metrics
print("\n[Logistic Regression Performance Metrics]")
print(f"Accuracy : {accuracy_score(y_test_c, y_pred_c):.4f}")
print(f"Precision: {precision_score(y_test_c, y_pred_c):.4f}")
print(f"Recall   : {recall_score(y_test_c, y_pred_c):.4f}")
print(f"F1-Score : {f1_score(y_test_c, y_pred_c):.4f}")

# Plot Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred_c)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Female', 'Male'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Gender Prediction')
plt.show()

# 2.4 Decision Boundary Visualization (ใช้ 2 Features หลัก)
# เลือก 2 Features ที่ใช้พล็อต เช่น Index 0 และ 1
feat1_idx, feat2_idx = 0, 1
X_2d = X_train_c_scaled[:, [feat1_idx, feat2_idx]]
y_2d = y_train_c.values

clf_2d = LogisticRegression()
clf_2d.fit(X_2d, y_2d)

# สร้าง Grid Mesh สำหรับวาด Decision Boundary
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap='coolwarm', edgecolors='k')
plt.xlabel(X_cls.columns[feat1_idx])
plt.ylabel(X_cls.columns[feat2_idx])
plt.title('Decision Boundary Visualization (Logistic Regression)')
plt.show()


# ==========================================
# LAB 3: MODEL COMPARISON
# ==========================================
print("\n" + "="*50)
print("LAB 3: MODEL COMPARISON & PERFORMANCE METRICS")
print("="*50)

# 3.1 Simple vs Multiple Linear Regression
reg_comparison = pd.DataFrame({
    'Model': ['Simple Linear Regression', 'Multiple Linear Regression'],
    'Test MSE': [mse_sim_test, mse_multi_test],
    'Test R²': [r2_sim_test, r2_multi_test]
})

print("\n--- 1. Simple vs Multiple Linear Regression ---")
print(reg_comparison.to_string(index=False))

# 3.2 Training vs Testing Performance (Overfitting Check)
train_vs_test = pd.DataFrame({
    'Dataset': ['Training Set', 'Testing Set'],
    'MSE': [mean_squared_error(y_train_r, y_pred_train_multi), mse_multi_test],
    'R² Score': [r2_score(y_train_r, y_pred_train_multi), r2_multi_test]
})

print("\n--- 2. Training vs Testing Performance (Multiple Regression) ---")
print(train_vs_test.to_string(index=False))

# 3.3 Summary Comparison: Regression vs Classification
summary_metrics = pd.DataFrame({
    'Task Type': ['Regression (Age)', 'Classification (Gender)'],
    'Primary Model': ['Multiple Linear Regression', 'Logistic Regression'],
    'Target Output': ['Continuous (Age in Years)', 'Binary Category (0/1)'],
    'Key Evaluation Metrics': ['MSE, RMSE, R² Score', 'Accuracy, F1-Score, Confusion Matrix']
})

print("\n--- 3. Regression vs Classification Overview ---")
print(summary_metrics.to_string(index=False))
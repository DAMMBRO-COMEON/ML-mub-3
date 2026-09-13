# ==========================================
# LAB : Support Vector Machine (SVM)
# Dataset : WineQT.csv
# ==========================================

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ==========================================
# 1. Load Dataset
# ==========================================
df = pd.read_csv(r"D:\Code\ML-CPE\ML-05\WineQT.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# ==========================================
# 2. Feature Selection
# ==========================================
X = df.drop(columns=["quality", "Id"])
y = df["quality"]

# ==========================================
# 3. Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# 4. Standardization
# ==========================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 5. SVM Models
# ==========================================
models = {
    "Linear": SVC(kernel='linear'),
    "Polynomial": SVC(kernel='poly', degree=3),
    "RBF": SVC(kernel='rbf')
}

# ==========================================
# 6. Train and Evaluate
# ==========================================
print("\n========== Accuracy Results ==========")

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print(f"{name} Kernel Accuracy = {accuracy:.4f}")

# ==========================================
# 7. Best Model
# ==========================================
best_kernel = max(results, key=results.get)

print("\n================================")
print("Best Kernel :", best_kernel)
print("Accuracy    :", round(results[best_kernel],4))
print("================================")

# ==========================================
# 8. Sample Predictions (Best Model)
# ==========================================
best_model = models[best_kernel]

predictions = best_model.predict(X_test)

comparison = pd.DataFrame({
    "Actual": y_test.values[:20],
    "Predicted": predictions[:20]
})

print("\nSample Predictions")
print(comparison)

# ==========================================
# 9. Summary Table
# ==========================================
result_df = pd.DataFrame({
    "Kernel": results.keys(),
    "Accuracy": results.values()
})

print("\nSummary")
print(result_df)

# ==========================================
# 10. Visualization : Compare 3 Kernels
# ==========================================

import matplotlib.pyplot as plt
import numpy as np

# ใช้ 2 Features สำหรับวาดกราฟ
feature_x = "alcohol"
feature_y = "volatile acidity"

X_plot = df[[feature_x, feature_y]]
y_plot = df["quality"]

# Standardize สำหรับกราฟ
plot_scaler = StandardScaler()
X_plot_scaled = plot_scaler.fit_transform(X_plot)

# สร้างโมเดล 3 Kernel สำหรับ Visualization
visual_models = {
    "Linear": SVC(kernel="linear"),
    "Polynomial": SVC(kernel="poly", degree=3),
    "RBF": SVC(kernel="rbf")
}

# สร้างพื้นที่สำหรับวาด Decision Boundary
x_min, x_max = X_plot_scaled[:, 0].min() - 1, X_plot_scaled[:, 0].max() + 1
y_min, y_max = X_plot_scaled[:, 1].min() - 1, X_plot_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# ==========================================
# สร้าง Figure เดียว แบ่งเป็น 3 กราฟ
# ==========================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, model) in zip(axes, visual_models.items()):

    # Train โมเดล
    model.fit(X_plot_scaled, y_plot)

    # ทำนายพื้นที่ในกราฟ
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    # แสดง Decision Boundary
    ax.contourf(
        xx,
        yy,
        Z,
        alpha=0.25
    )

    # แสดงข้อมูลจริง
    ax.scatter(
        X_plot_scaled[:, 0],
        X_plot_scaled[:, 1],
        c=y_plot,
        edgecolors="k"
    )

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title(f"{name} Kernel")
    ax.grid()

plt.suptitle("Comparison of SVM Kernels", fontsize=16)

plt.tight_layout()
plt.show()
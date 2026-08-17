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
import io
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"D:\Code\ML-CPE\ML-05\WineQT.csv")

# ตัดแถวที่มีค่าว่าง (NaN) ในคอลัมน์สำคัญออกให้หมด
df = df.dropna(subset=["quality_encoded"])

X = df.drop(columns=["quality", "quality_encoded", "Id"])
y = df["quality_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_values = [3, 5, 7]
accuracies = {}

print("--- Model Evaluation ---")
for k in k_values:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train_scaled, y_train)
  y_pred = knn.predict(X_test_scaled)
  acc = accuracy_score(y_test, y_pred)
  accuracies[k] = acc
  print(f"KNN with k = {k} | Accuracy: {acc * 100:.2f}%")

best_k = max(accuracies, key=accuracies.get)

print("\n" + "=" * 40)
print("OUTPUT RESULTS")
print("=" * 40)
print(f"Accuracy scores for each k value: {accuracies}")
print(f"The best k value based on test accuracy: k = {best_k}")
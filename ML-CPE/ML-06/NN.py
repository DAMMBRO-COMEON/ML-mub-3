
# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(r"D:\Code\ML-CPE\ML-06\WineQT.csv")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 3. PREPARE DATA
# ============================================================

# X = Features
# ไม่ใช้ Id เพราะเป็นเพียงหมายเลขประจำข้อมูล
X = df.drop(columns=["quality", "Id"])

# y = Target
y = df["quality"]


# ============================================================
# 4. ENCODE TARGET
# ============================================================

# เปลี่ยน quality 3,4,5,6,7,8
# ให้เป็น 0,1,2,3,4,5

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("\nClasses:")
print(encoder.classes_)


# ============================================================
# 5. SPLIT TRAINING / TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)


# ============================================================
# 6. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nStandardization completed.")


# ============================================================
# 7. FUNCTION FOR CREATING NEURAL NETWORK
# ============================================================

def create_model(hidden_layers):

    model = Sequential()

    # Input + first hidden layer
    model.add(
        Dense(
            hidden_layers[0],
            activation="relu",
            input_shape=(X_train.shape[1],)
        )
    )

    # Additional hidden layers
    for neurons in hidden_layers[1:]:
        model.add(
            Dense(
                neurons,
                activation="relu"
            )
        )

    # Output layer
    model.add(
        Dense(
            len(np.unique(y)),
            activation="softmax"
        )
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 8. EXPERIMENT 1
# Compare Different Numbers of Epochs
# ============================================================

epochs_list = [20, 50, 100]

epoch_results = []

print("\n")
print("=" * 60)
print("EXPERIMENT 1: DIFFERENT EPOCHS")
print("=" * 60)

for epochs in epochs_list:

    model = create_model([16, 8])

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.20,
        verbose=0
    )

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    epoch_results.append(
        [epochs, accuracy]
    )

    print(
        f"Epochs: {epochs:3d} | "
        f"Test Accuracy: {accuracy:.4f}"
    )


# ============================================================
# 9. SHOW EPOCH COMPARISON
# ============================================================

epoch_df = pd.DataFrame(
    epoch_results,
    columns=["Epochs", "Accuracy"]
)

print("\nEpoch Comparison:")
print(epoch_df)


# ============================================================
# 10. EXPERIMENT 2
# Compare Different Neural Network Configurations
# ============================================================

configs = {
    "NN-1": [16],
    "NN-2": [16, 8],
    "NN-3": [64, 32, 16]
}

config_results = []

print("\n")
print("=" * 60)
print("EXPERIMENT 2: DIFFERENT NN CONFIGURATIONS")
print("=" * 60)

for name, layers in configs.items():

    model = create_model(layers)

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.20,
        verbose=0
    )

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    config_results.append(
        [name, str(layers), accuracy]
    )

    print(
        f"{name} | "
        f"Layers: {layers} | "
        f"Accuracy: {accuracy:.4f}"
    )


# ============================================================
# 11. SHOW CONFIGURATION COMPARISON
# ============================================================

config_df = pd.DataFrame(
    config_results,
    columns=["Model", "Hidden Layers", "Accuracy"]
)

print("\nNN Configuration Comparison:")
print(config_df)


# ============================================================
# 12. TRAIN FINAL MODEL
# ============================================================

# ใช้ Configuration ที่เลือก
# [16, 8] = Hidden Layer 1 มี 16 neurons
#           Hidden Layer 2 มี 8 neurons

final_model = create_model([16, 8])

history = final_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.20,
    verbose=1
)


# ============================================================
# 13. FINAL MODEL EVALUATION
# ============================================================

test_loss, test_accuracy = final_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n")
print("=" * 60)
print("FINAL MODEL RESULT")
print("=" * 60)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# 14. PREDICTION
# ============================================================

y_pred_probability = final_model.predict(
    X_test,
    verbose=0
)

# เลือก Class ที่มี Probability สูงที่สุด
y_pred = np.argmax(
    y_pred_probability,
    axis=1
)

# แปลงกลับจาก 0,1,2,3,4,5
# เป็น quality 3,4,5,6,7,8

y_pred_original = encoder.inverse_transform(y_pred)
y_test_original = encoder.inverse_transform(y_test)


print("\nPredictions:")
print(y_pred_original[:20])

print("\nActual:")
print(y_test_original[:20])


# ============================================================
# 15. ACCURACY USING SKLEARN
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nPrediction Accuracy:")
print(f"{accuracy * 100:.2f}%")


# ============================================================
# 16. TRAINING / VALIDATION ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()
plt.grid()

plt.show()


# ============================================================
# 17. TRAINING / VALIDATION LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()
plt.grid()

plt.show()


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("LAB SUMMARY")
print("=" * 60)

print("\nEpoch Comparison:")
print(epoch_df)

print("\nNN Configuration Comparison:")
print(config_df)

print("\nFinal Model Accuracy:")
print(f"{test_accuracy * 100:.2f}%")
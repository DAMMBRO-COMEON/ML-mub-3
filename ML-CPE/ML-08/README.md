"""
Lab 8: Deep Convolutional Neural Network (DCNN) and Applications
Dataset: WineQT.csv
Description: Complete Python script for data preprocessing, 1D-CNN model building,
training with various configurations and epochs, evaluation, and visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("--- 1. Loading and Exploring Dataset ---")
# Load dataset (Make sure WineQT.csv is in the same directory)
try:
    df = pd.read_csv('WineQT.csv')
except FileNotFoundError:
    print("Error: WineQT.csv not found. Please ensure the dataset file is uploaded.")
    exit()

print(f"Dataset shape: {df.shape}")
print(df.head())

# Prepare features and target
# Drop 'Id' if present
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Binarize target: quality >= 6 is Good (1), else 0
df['quality_binary'] = (df['quality'] >= 6).astype(int)

X = df.drop(columns=['quality', 'quality_binary']).values
y = df['quality_binary'].values

print(f"Features shape: {X.shape}, Target shape: {y.shape}")
print(f"Class distribution:\n{pd.Series(y).value_counts(normalize=True)}")

print("\n--- 2. Data Preprocessing & Splitting ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for 1D-CNN (Samples, Timesteps/Features, Channels=1)
X_train_reshaped = np.expand_dims(X_train_scaled, axis=2)
X_test_reshaped = np.expand_dims(X_test_scaled, axis=2)

print(f"Reshaped Training data shape: {X_train_reshaped.shape}")
print(f"Reshaped Testing data shape: {X_test_reshaped.shape}")


# Function to build DCNN models
def build_dcnn_model(config_type, input_shape):
    model = Sequential()
    if config_type == 'Config_A':
        # 1 Conv Layer, 32 filters
        model.add(
            Conv1D(
                32,
                kernel_size=3,
                activation='relu',
                input_shape=input_shape,
                padding='same',
            )
        )
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    elif config_type == 'Config_B':
        # 1 Conv Layer, 64 filters
        model.add(
            Conv1D(
                64,
                kernel_size=3,
                activation='relu',
                input_shape=input_shape,
                padding='same',
            )
        )
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    elif config_type == 'Config_C':
        # 2 Conv Layers
        model.add(
            Conv1D(
                32,
                kernel_size=3,
                activation='relu',
                input_shape=input_shape,
                padding='same',
            )
        )
        model.add(
            Conv1D(64, kernel_size=3, activation='relu', padding='same')
        )
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


print("\n--- 3. Model Training & Evaluation Across Configurations & Epochs ---")
configurations = ['Config_A', 'Config_B', 'Config_C']
epoch_options = [10, 25, 50]
results = []

input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])

for config in configurations:
    for epochs in epoch_options:
        print(
            f"\nTraining Model: {config} with {epochs} epochs..."
        )
        model = build_dcnn_model(config, input_shape)

        history = model.fit(
            X_train_reshaped,
            y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
        )

        # Evaluate on test set
        test_loss, test_acc = model.evaluate(
            X_test_reshaped, y_test, verbose=0
        )
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]

        print(
            f"  -> Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}"
        )

        results.append({
            'Configuration': config,
            'Epochs': epochs,
            'Train Accuracy': train_acc,
            'Val Accuracy': val_acc,
            'Test Accuracy': test_acc,
            'Test Loss': test_loss,
            'Model': model,
            'History': history,
        })

# Summary DataFrame
results_df = pd.DataFrame([{
    'Configuration': r['Configuration'],
    'Epochs': r['Epochs'],
    'Train Accuracy': f"{r['Train Accuracy']*100:.2f}%",
    'Val Accuracy': f"{r['Val Accuracy']*100:.2f}%",
    'Test Accuracy': f"{r['Test Accuracy']*100:.2f}%",
    'Test Loss': f"{r['Test Loss']:.4f}",
} for r in results])

print("\n--- Summary of All Experiments ---")
print(results_df.to_string(index=False))

print("\n--- 4. Detailed Evaluation of Best Model ---")
# Find best model based on test accuracy
best_result = max(results, key=lambda x: x['Test Accuracy'])
print(
    f"\nBest Performing Model: {best_result['Configuration']} with {best_result['Epochs']} epochs"
)
print(f"Test Accuracy: {best_result['Test Accuracy']*100:.2f}%")

best_model = best_result['Model']
y_pred_prob = best_model.predict(X_test_reshaped)
y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nSample Predictions (First 5 test samples):")
for i in range(5):
    actual = "Good (1)" if y_test[i] == 1 else "Normal (0)"
    pred = "Good (1)" if y_pred[i] == 1 else "Normal (0)"
    conf = (
        y_pred_prob[i][0] if y_pred[i] == 1 else (1 - y_pred_prob[i][0])
    ) * 100
    print(
        f"  Example {i+1}: Actual = {actual} | Predicted = {pred} [Confidence: {conf:.2f}%]"
    )

print("\n--- Lab Execution Complete ---")
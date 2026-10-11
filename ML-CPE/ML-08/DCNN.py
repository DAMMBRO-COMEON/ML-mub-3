import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense

# กำหนด Seed เพื่อให้ผลลัพธ์คงที่และสามารถทำซ้ำได้เหมือนกันทุกครั้ง
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=== (Loading & Preprocessing) ===")
# โหลดชุดข้อมูล WineQT.csv
df = pd.read_csv(r"D:\Code\ML-CPE\ML-08\WineQT.csv")
print(f"ขนาดข้อมูลทั้งหมด: {df.shape}")

# แยก Features (X) และ Label (y)
X = df.drop(['quality', 'Id'], axis=1).values
y = df['quality'].values

# แปลง Label เป็น Binary Classification (คุณภาพดีเกรด >= 6 ให้เป็น 1, นอกนั้นเป็น 0)
y_binary = (y >= 6).astype(int)

# แบ่งข้อมูลเป็น Training Set (80%) และ Test Set (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=RANDOM_SEED, stratify=y_binary
)

# ทำมาตรฐานข้อมูล (Standardization) ให้มี Mean = 0 และ Variance = 1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ปรับรูปทรงข้อมูลให้เหมาะสมกับ Conv1D (Samples, Timesteps, Features) = (N, 11, 1)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

print(f"(Train Shape): {X_train_reshaped.shape}")
print(f"(Test Shape): {X_test_reshaped.shape}\n")


# ฟังก์ชันสร้างโมเดล DCNN ตามโครงสร้าง (Configuration) ต่างๆ
def create_dcnn_model(config_type):
    model = Sequential()
    
    if config_type == 'Config A (1 Conv Layer, 32 filters)':
        model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(11, 1)))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
    elif config_type == 'Config B (1 Conv Layer, 64 filters)':
        model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(11, 1)))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
    elif config_type == 'Config C (2 Conv Layers)':
        model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(11, 1)))
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


print("===(Training & Evaluation) ===")
configs = [
    'Config A (1 Conv Layer, 32 filters)',
    'Config B (1 Conv Layer, 64 filters)',
    'Config C (2 Conv Layers)'
]
epochs_list = [10, 25, 50]

results = []
best_history = None
best_fitted_model = None
best_test_acc = 0

for cfg in configs:
    for ep in epochs_list:
        print(f"กำลังฝึกสอน: {cfg} | Epochs: {ep}")
        model = create_dcnn_model(cfg)
        
        # ฝึกสอนโมเดลโดยแบ่ง Validation Split 20% จากชุด Train
        history = model.fit(
            X_train_reshaped, y_train,
            epochs=ep,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # ประเมินผลบน Test Set
        test_loss, test_acc = model.evaluate(X_test_reshaped, y_test, verbose=0)
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        results.append({
            'Configuration': cfg,
            'Epochs': ep,
            'Train Accuracy (%)': round(train_acc * 100, 2),
            'Val Accuracy (%)': round(val_acc * 100, 2),
            'Test Accuracy (%)': round(test_acc * 100, 2),
            'Test Loss': round(test_loss, 4)
        })
        
        # บันทึกโมเดลและ history ที่ให้ผลดีที่สุดไว้สำหรับพล็อตและทำนาย
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_history = history
            best_fitted_model = model
            best_config_name = f"{cfg} ({ep} Epochs)"

print("\n=== (Summary Table) ===")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))


print(f"\n=== (Sample Predictions) โดยใช้โมเดลที่ดีที่สุด: {best_config_name} ===")
sample_X = X_test_reshaped[:5]
sample_y_true = y_test[:5]
predictions = best_fitted_model.predict(sample_X)

for i in range(5):
    prob = predictions[i][0]
    pred_label = 1 if prob >= 0.5 else 0
    confidence = prob if pred_label == 1 else (1 - prob)
    true_label = sample_y_true[i]
    
    print(f"ตัวอย่างที่ {i+1}: ค่าจริง = {true_label} | พยากรณ์ = {pred_label} | ความมั่นใจ = {confidence*100:.2f}%")


print("\n=== (Plotting & Saving Graphs) ===")
plt.figure(figsize=(14, 5))

# กราฟแสดงความแม่นยำ (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(best_history.history['accuracy'], label='Train Accuracy', linewidth=2, color='blue')
plt.plot(best_history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='orange')
plt.title(f'Model Accuracy\n{best_config_name}', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

# กราฟแสดงค่าความสูญเสีย (Loss)
plt.subplot(1, 2, 2)
plt.plot(best_history.history['loss'], label='Train Loss', linewidth=2, color='blue')
plt.plot(best_history.history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
plt.title(f'Model Loss\n{best_config_name}', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300)
print("บันทึกกราฟเรียบร้อยแล้วในชื่อไฟล์ 'training_history.png'")
plt.show()

print("\nการทำงานเสร็จสมบูรณ์!")
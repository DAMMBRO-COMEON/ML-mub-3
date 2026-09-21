# ================================================================
# LAB: CNN on a Dataset of Your Choice
# Dataset: WineQT.csv
# Task: Wine Quality Classification using 1D CNN
#
# Output:
# 1. Accuracy scores for each CNN configuration
# 2. Comparison using different epochs
# 3. Training / Validation Accuracy
# 4. Training / Validation Loss
# 5. Predictions
# 6. Confusion Matrix
# 7. Save ALL graphs as PNG
# 8. Open ALL graphs after program finishes
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import os

# ลดข้อความ Log ของ TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# ================================================================
# 2. RANDOM SEED
# ================================================================

np.random.seed(42)
tf.random.set_seed(42)


# ================================================================
# 3. FILE PATH
# ================================================================
# หา Folder ที่ CNN.py อยู่
# แล้วใช้ WineQT.csv จาก Folder เดียวกัน
# ================================================================

current_folder = os.path.dirname(
    os.path.abspath(__file__)
)

csv_path = os.path.join(
    current_folder,
    "WineQT.csv"
)


# ================================================================
# 4. CREATE GRAPH FOLDER
# ================================================================

graph_folder = os.path.join(
    current_folder,
    "CNN_Graphs"
)

os.makedirs(
    graph_folder,
    exist_ok=True
)


# ================================================================
# 5. LOAD DATASET
# ================================================================

print("=" * 80)
print("LOAD DATASET")
print("=" * 80)

print("\nPython Folder:")
print(current_folder)

print("\nDataset Path:")
print(csv_path)


if not os.path.exists(csv_path):

    print("\nERROR: WineQT.csv not found!")

    print("\nPlease put WineQT.csv in:")
    print(current_folder)

    input("\nPress Enter to exit...")

    raise SystemExit


df = pd.read_csv(csv_path)

print("\nDataset loaded successfully!")


print("\nFirst 5 rows:")
print(df.head())


print("\nDataset Shape:")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())


# ================================================================
# 6. CHECK MISSING VALUES
# ================================================================

print("\n")
print("=" * 80)
print("MISSING VALUES")
print("=" * 80)

print(
    df.isnull().sum()
)


# ================================================================
# 7. QUALITY DISTRIBUTION
# ================================================================

print("\n")
print("=" * 80)
print("QUALITY DISTRIBUTION")
print("=" * 80)


quality_counts = (
    df["quality"]
    .value_counts()
    .sort_index()
)


print(
    quality_counts
)


# ================================================================
# GRAPH 1
# WINE QUALITY DISTRIBUTION
# ================================================================

plt.figure(
    figsize=(8, 5)
)


bars = plt.bar(
    quality_counts.index.astype(str),
    quality_counts.values,
    color="steelblue"
)


plt.title(
    "Wine Quality Distribution"
)

plt.xlabel(
    "Wine Quality"
)

plt.ylabel(
    "Number of Samples"
)

plt.grid(
    axis="y",
    alpha=0.3
)


# ตัวเลขบน Bar
for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 2,
        f"{int(height)}",
        ha="center"
    )


plt.tight_layout()


graph1 = os.path.join(
    graph_folder,
    "01_quality_distribution.png"
)


plt.savefig(
    graph1,
    dpi=300,
    bbox_inches="tight"
)


# ไม่ใช้ plt.show() ตรงนี้
# เพื่อไม่ให้โปรแกรมหยุดรอ
plt.close()


# ================================================================
# 8. FEATURES AND TARGET
# ================================================================
#
# quality = Target
# Id      = ไม่ใช้เป็น Feature
#
# ================================================================

X = df.drop(
    columns=[
        "quality",
        "Id"
    ]
)


y = df["quality"]


print("\n")
print("=" * 80)
print("FEATURES")
print("=" * 80)


for number, feature in enumerate(
    X.columns,
    start=1
):

    print(
        f"{number:2d}. {feature}"
    )


print(
    "\nNumber of Features:",
    X.shape[1]
)


print(
    "Target: quality"
)


# ================================================================
# 9. LABEL ENCODING
# ================================================================

label_encoder = LabelEncoder()


y_encoded = label_encoder.fit_transform(
    y
)


number_of_classes = len(
    label_encoder.classes_
)


print("\n")
print("=" * 80)
print("LABEL ENCODING")
print("=" * 80)


for encoded_class, quality_class in enumerate(
    label_encoder.classes_
):

    print(
        f"Class {encoded_class} -> "
        f"Quality {quality_class}"
    )


print(
    "\nNumber of Classes:",
    number_of_classes
)


# ================================================================
# 10. TRAIN / TEST SPLIT
# ================================================================
#
# Training 80%
# Testing 20%
#
# stratify:
# รักษาสัดส่วนแต่ละ Quality
#
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded
)


print("\n")
print("=" * 80)
print("TRAIN / TEST SPLIT")
print("=" * 80)


print(
    f"Total Samples    : {len(df)}"
)

print(
    f"Training Samples : {len(X_train)}"
)

print(
    f"Testing Samples  : {len(X_test)}"
)


# ================================================================
# 11. STANDARDIZATION
# ================================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)


print("\n")
print("=" * 80)
print("STANDARDIZATION")
print("=" * 80)


print("\nBefore:")
print(
    X_train.iloc[0].values
)


print("\nAfter:")
print(
    X_train_scaled[0]
)


# ================================================================
# 12. RESHAPE FOR CNN
# ================================================================
#
# Original:
#
# (samples, 11)
#
# CNN:
#
# (samples, 11, 1)
#
# ================================================================

X_train_cnn = X_train_scaled.reshape(

    X_train_scaled.shape[0],

    X_train_scaled.shape[1],

    1
)


X_test_cnn = X_test_scaled.reshape(

    X_test_scaled.shape[0],

    X_test_scaled.shape[1],

    1
)


print("\n")
print("=" * 80)
print("CNN INPUT SHAPE")
print("=" * 80)


print(
    "Training:",
    X_train_cnn.shape
)

print(
    "Testing :",
    X_test_cnn.shape
)


# ================================================================
# 13. ONE-HOT ENCODING
# ================================================================

y_train_cat = to_categorical(

    y_train,

    num_classes=number_of_classes
)


y_test_cat = to_categorical(

    y_test,

    num_classes=number_of_classes
)


# ================================================================
# 14. CREATE CNN MODEL
# ================================================================

def create_cnn_model(config):

    model = Sequential()


    # ============================================================
    # CNN 1
    #
    # Conv1D 32
    # Dense 32
    # ============================================================

    if config == 1:

        model.add(
            Input(
                shape=(
                    X_train_cnn.shape[1],
                    1
                )
            )
        )


        model.add(
            Conv1D(
                filters=32,
                kernel_size=3,
                activation="relu",
                padding="same"
            )
        )


        model.add(
            MaxPooling1D(
                pool_size=2
            )
        )


        model.add(
            Flatten()
        )


        model.add(
            Dense(
                32,
                activation="relu"
            )
        )


        model.add(
            Dropout(
                0.30
            )
        )


        model.add(
            Dense(
                number_of_classes,
                activation="softmax"
            )
        )


    # ============================================================
    # CNN 2
    #
    # Conv1D 32
    # Conv1D 64
    # Dense 64
    # ============================================================

    elif config == 2:

        model.add(
            Input(
                shape=(
                    X_train_cnn.shape[1],
                    1
                )
            )
        )


        model.add(
            Conv1D(
                filters=32,
                kernel_size=3,
                activation="relu",
                padding="same"
            )
        )


        model.add(
            Conv1D(
                filters=64,
                kernel_size=3,
                activation="relu",
                padding="same"
            )
        )


        model.add(
            MaxPooling1D(
                pool_size=2
            )
        )


        model.add(
            Flatten()
        )


        model.add(
            Dense(
                64,
                activation="relu"
            )
        )


        model.add(
            Dropout(
                0.30
            )
        )


        model.add(
            Dense(
                number_of_classes,
                activation="softmax"
            )
        )


    # ============================================================
    # CNN 3
    #
    # Conv1D 64
    # Conv1D 128
    # Dense 128
    # Dense 64
    # ============================================================

    elif config == 3:

        model.add(
            Input(
                shape=(
                    X_train_cnn.shape[1],
                    1
                )
            )
        )


        model.add(
            Conv1D(
                filters=64,
                kernel_size=3,
                activation="relu",
                padding="same"
            )
        )


        model.add(
            Conv1D(
                filters=128,
                kernel_size=3,
                activation="relu",
                padding="same"
            )
        )


        model.add(
            MaxPooling1D(
                pool_size=2
            )
        )


        model.add(
            Flatten()
        )


        model.add(
            Dense(
                128,
                activation="relu"
            )
        )


        model.add(
            Dropout(
                0.30
            )
        )


        model.add(
            Dense(
                64,
                activation="relu"
            )
        )


        model.add(
            Dropout(
                0.20
            )
        )


        model.add(
            Dense(
                number_of_classes,
                activation="softmax"
            )
        )


    else:

        raise ValueError(
            "Configuration must be 1, 2 or 3"
        )


    # ============================================================
    # COMPILE
    # ============================================================

    model.compile(

        optimizer=Adam(
            learning_rate=0.001
        ),

        loss="categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )


    return model


# ================================================================
# 15. CONFIGURATION
# ================================================================

epoch_list = [
    20,
    50,
    100
]


configs = [
    1,
    2,
    3
]


config_names = {

    1:
        "CNN 1: Conv32 + Dense32",

    2:
        "CNN 2: Conv32-64 + Dense64",

    3:
        "CNN 3: Conv64-128 + Dense128-64"
}


# ================================================================
# 16. VARIABLES
# ================================================================

results = []

histories = {}

trained_models = {}


# ================================================================
# 17. TRAIN CNN
# ================================================================

print("\n")
print("=" * 80)
print("START TRAINING")
print("=" * 80)


experiment_number = 1


for config in configs:

    for epochs in epoch_list:


        print("\n")
        print("=" * 80)

        print(
            f"Experiment {experiment_number}/9"
        )

        print(
            f"Model  : {config_names[config]}"
        )

        print(
            f"Epochs : {epochs}"
        )

        print("=" * 80)


        # --------------------------------------------------------
        # Clear Session
        # --------------------------------------------------------

        tf.keras.backend.clear_session()


        # --------------------------------------------------------
        # Create Model
        # --------------------------------------------------------

        model = create_cnn_model(
            config
        )


        # --------------------------------------------------------
        # Train
        # --------------------------------------------------------

        history = model.fit(

            X_train_cnn,

            y_train_cat,

            epochs=epochs,

            batch_size=32,

            validation_split=0.20,

            verbose=0
        )


        # --------------------------------------------------------
        # Test
        # --------------------------------------------------------

        test_loss, test_accuracy = model.evaluate(

            X_test_cnn,

            y_test_cat,

            verbose=0
        )


        train_accuracy = history.history[
            "accuracy"
        ][-1]


        validation_accuracy = history.history[
            "val_accuracy"
        ][-1]


        train_loss = history.history[
            "loss"
        ][-1]


        validation_loss = history.history[
            "val_loss"
        ][-1]


        # --------------------------------------------------------
        # Print
        # --------------------------------------------------------

        print(
            f"Training Accuracy   : "
            f"{train_accuracy * 100:.2f}%"
        )


        print(
            f"Validation Accuracy : "
            f"{validation_accuracy * 100:.2f}%"
        )


        print(
            f"Test Accuracy       : "
            f"{test_accuracy * 100:.2f}%"
        )


        print(
            f"Training Loss       : "
            f"{train_loss:.4f}"
        )


        print(
            f"Validation Loss     : "
            f"{validation_loss:.4f}"
        )


        print(
            f"Test Loss           : "
            f"{test_loss:.4f}"
        )


        # --------------------------------------------------------
        # Save Result
        # --------------------------------------------------------

        results.append(
            {
                "Configuration":
                    config_names[config],

                "Config ID":
                    config,

                "Epochs":
                    epochs,

                "Training Accuracy (%)":
                    train_accuracy * 100,

                "Validation Accuracy (%)":
                    validation_accuracy * 100,

                "Test Accuracy (%)":
                    test_accuracy * 100,

                "Test Accuracy":
                    test_accuracy,

                "Training Loss":
                    train_loss,

                "Validation Loss":
                    validation_loss,

                "Test Loss":
                    test_loss
            }
        )


        # --------------------------------------------------------
        # Save History
        # --------------------------------------------------------

        histories[
            (config, epochs)
        ] = history


        # --------------------------------------------------------
        # Save Model
        # --------------------------------------------------------

        trained_models[
            (config, epochs)
        ] = model


        experiment_number += 1


# ================================================================
# 18. RESULT DATAFRAME
# ================================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 130)
print("ALL CNN RESULTS")
print("=" * 130)


print(
    results_df[
        [
            "Configuration",
            "Epochs",
            "Training Accuracy (%)",
            "Validation Accuracy (%)",
            "Test Accuracy (%)",
            "Test Loss"
        ]
    ].round(4).to_string(
        index=False
    )
)


# ================================================================
# 19. FIND BEST MODEL
# ================================================================

best_index = results_df[
    "Test Accuracy"
].idxmax()


best_result = results_df.loc[
    best_index
]


best_config = int(
    best_result["Config ID"]
)


best_epoch = int(
    best_result["Epochs"]
)


best_accuracy = float(
    best_result["Test Accuracy (%)"]
)


best_loss = float(
    best_result["Test Loss"]
)


best_model = trained_models[
    (
        best_config,
        best_epoch
    )
]


best_history = histories[
    (
        best_config,
        best_epoch
    )
]


print("\n")
print("=" * 80)
print("BEST MODEL")
print("=" * 80)


print(
    "Configuration :",
    config_names[best_config]
)


print(
    "Epochs        :",
    best_epoch
)


print(
    f"Accuracy      : "
    f"{best_accuracy:.2f}%"
)


print(
    f"Loss          : "
    f"{best_loss:.4f}"
)


# ================================================================
# GRAPH 2
# TRAINING AND VALIDATION ACCURACY
# ================================================================

plt.figure(
    figsize=(10, 6)
)


epoch_range = range(
    1,
    len(
        best_history.history["accuracy"]
    ) + 1
)


plt.plot(
    epoch_range,
    best_history.history["accuracy"],
    linewidth=2,
    label="Training Accuracy"
)


plt.plot(
    epoch_range,
    best_history.history["val_accuracy"],
    linewidth=2,
    label="Validation Accuracy"
)


plt.title(
    "Training and Validation Accuracy\n"
    f"{config_names[best_config]} | "
    f"{best_epoch} Epochs"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


graph2 = os.path.join(
    graph_folder,
    "02_training_validation_accuracy.png"
)


plt.savefig(
    graph2,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ================================================================
# GRAPH 3
# TRAINING AND VALIDATION LOSS
# ================================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    epoch_range,
    best_history.history["loss"],
    linewidth=2,
    label="Training Loss"
)


plt.plot(
    epoch_range,
    best_history.history["val_loss"],
    linewidth=2,
    label="Validation Loss"
)


plt.title(
    "Training and Validation Loss\n"
    f"{config_names[best_config]} | "
    f"{best_epoch} Epochs"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


graph3 = os.path.join(
    graph_folder,
    "03_training_validation_loss.png"
)


plt.savefig(
    graph3,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ================================================================
# GRAPH 4
# EPOCH VS TEST ACCURACY
# ================================================================

plt.figure(
    figsize=(11, 7)
)


for config in configs:

    temp = results_df[
        results_df["Config ID"] == config
    ].sort_values(
        "Epochs"
    )


    plt.plot(

        temp["Epochs"],

        temp["Test Accuracy (%)"],

        marker="o",

        markersize=8,

        linewidth=2,

        label=config_names[config]
    )


    # ตัวเลข Accuracy แต่ละจุด
    for epoch_value, accuracy_value in zip(
        temp["Epochs"],
        temp["Test Accuracy (%)"]
    ):

        plt.annotate(

            f"{accuracy_value:.2f}%",

            (
                epoch_value,
                accuracy_value
            ),

            textcoords="offset points",

            xytext=(0, 8),

            ha="center",

            fontsize=8
        )


plt.title(
    "Test Accuracy Comparison with Different Epochs"
)


plt.xlabel(
    "Number of Epochs"
)


plt.ylabel(
    "Test Accuracy (%)"
)


plt.xticks(
    epoch_list
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


graph4 = os.path.join(
    graph_folder,
    "04_epoch_vs_accuracy.png"
)


plt.savefig(
    graph4,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ================================================================
# GRAPH 5
# CNN CONFIGURATION COMPARISON
# ================================================================

plt.figure(
    figsize=(12, 7)
)


x = np.arange(
    len(configs)
)


width = 0.25


for index, epochs in enumerate(
    epoch_list
):

    accuracy_values = []


    for config in configs:

        current_result = results_df[
            (results_df["Config ID"] == config)
            &
            (results_df["Epochs"] == epochs)
        ]


        accuracy_value = float(
            current_result[
                "Test Accuracy (%)"
            ].iloc[0]
        )


        accuracy_values.append(
            accuracy_value
        )


    positions = (
        x
        +
        (index - 1) * width
    )


    bars = plt.bar(

        positions,

        accuracy_values,

        width,

        label=f"{epochs} Epochs"
    )


    for bar, accuracy_value in zip(
        bars,
        accuracy_values
    ):

        plt.text(

            bar.get_x()
            +
            bar.get_width() / 2,

            bar.get_height()
            +
            0.2,

            f"{accuracy_value:.1f}%",

            ha="center",

            va="bottom",

            fontsize=8
        )


plt.xticks(
    x,
    [
        "CNN 1",
        "CNN 2",
        "CNN 3"
    ]
)


plt.xlabel(
    "CNN Configuration"
)


plt.ylabel(
    "Test Accuracy (%)"
)


plt.title(
    "CNN Configuration Accuracy Comparison"
)


plt.legend()


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


graph5 = os.path.join(
    graph_folder,
    "05_cnn_configuration_comparison.png"
)


plt.savefig(
    graph5,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ================================================================
# 20. PREDICTIONS
# ================================================================

prediction_probability = best_model.predict(
    X_test_cnn,
    verbose=0
)


predicted_class = np.argmax(
    prediction_probability,
    axis=1
)


predicted_quality = label_encoder.inverse_transform(
    predicted_class
)


actual_quality = label_encoder.inverse_transform(
    y_test
)


prediction_df = pd.DataFrame(
    {
        "Actual Quality":
            actual_quality,

        "Predicted Quality":
            predicted_quality,

        "Correct":
            actual_quality == predicted_quality
    }
)


# ================================================================
# 21. SHOW PREDICTIONS
# ================================================================

print("\n")
print("=" * 80)
print("FIRST 30 PREDICTIONS")
print("=" * 80)


print(
    prediction_df.head(
        30
    ).to_string(
        index=False
    )
)


# ================================================================
# 22. PREDICTION ACCURACY
# ================================================================

final_accuracy = accuracy_score(
    actual_quality,
    predicted_quality
)


correct_predictions = int(
    np.sum(
        actual_quality
        ==
        predicted_quality
    )
)


wrong_predictions = int(
    np.sum(
        actual_quality
        !=
        predicted_quality
    )
)


print("\n")
print("=" * 80)
print("PREDICTION RESULT")
print("=" * 80)


print(
    f"Correct : "
    f"{correct_predictions}"
)


print(
    f"Wrong   : "
    f"{wrong_predictions}"
)


print(
    f"Total   : "
    f"{len(actual_quality)}"
)


print(
    f"Accuracy: "
    f"{final_accuracy * 100:.2f}%"
)


# ================================================================
# GRAPH 6
# ACTUAL VS PREDICTED
# ================================================================

number_to_show = min(
    50,
    len(actual_quality)
)


sample_numbers = np.arange(
    1,
    number_to_show + 1
)


plt.figure(
    figsize=(14, 7)
)


plt.plot(

    sample_numbers,

    actual_quality[
        :number_to_show
    ],

    marker="o",

    markersize=6,

    linewidth=2,

    label="Actual Quality"
)


plt.plot(

    sample_numbers,

    predicted_quality[
        :number_to_show
    ],

    marker="x",

    markersize=7,

    linewidth=2,

    label="Predicted Quality"
)


plt.title(
    "Actual vs Predicted Wine Quality"
)


plt.xlabel(
    "Test Sample"
)


plt.ylabel(
    "Wine Quality"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


graph6 = os.path.join(
    graph_folder,
    "06_actual_vs_predicted.png"
)


plt.savefig(
    graph6,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ================================================================
# 23. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(

    actual_quality,

    predicted_quality,

    labels=label_encoder.classes_
)


print("\n")
print("=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(cm)


# ================================================================
# GRAPH 7
# CONFUSION MATRIX
# ================================================================

plt.figure(
    figsize=(8, 7)
)


plt.imshow(
    cm,
    interpolation="nearest",
    cmap="Blues"
)


plt.title(
    "Confusion Matrix"
)


plt.colorbar()


tick_marks = np.arange(
    len(
        label_encoder.classes_
    )
)


plt.xticks(
    tick_marks,
    label_encoder.classes_
)


plt.yticks(
    tick_marks,
    label_encoder.classes_
)


plt.xlabel(
    "Predicted Quality"
)


plt.ylabel(
    "Actual Quality"
)


threshold = (
    cm.max() / 2.0
    if cm.max() > 0
    else 0
)


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(

            j,

            i,

            str(
                cm[i, j]
            ),

            ha="center",

            va="center",

            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            )
        )


plt.tight_layout()


graph7 = os.path.join(
    graph_folder,
    "07_confusion_matrix.png"
)


plt.savefig(
    graph7,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ================================================================
# 24. CLASSIFICATION REPORT
# ================================================================

print("\n")
print("=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)


print(
    classification_report(

        actual_quality,

        predicted_quality,

        labels=label_encoder.classes_,

        zero_division=0
    )
)


# ================================================================
# 25. SUMMARY ALL EXPERIMENTS
# ================================================================

print("\n")
print("=" * 110)
print("SUMMARY OF ALL EXPERIMENTS")
print("=" * 110)


for _, row in results_df.iterrows():

    configuration_name = str(
        row["Configuration"]
    )


    epochs_value = int(
        row["Epochs"]
    )


    accuracy_value = float(
        row["Test Accuracy (%)"]
    )


    loss_value = float(
        row["Test Loss"]
    )


    print(

        f"{configuration_name:<40} | "

        f"Epochs = {epochs_value:3d} | "

        f"Accuracy = {accuracy_value:6.2f}% | "

        f"Loss = {loss_value:.4f}"
    )


# ================================================================
# 26. BEST RESULT EACH CNN
# ================================================================

print("\n")
print("=" * 110)
print("BEST RESULT OF EACH CNN CONFIGURATION")
print("=" * 110)


best_cnn_names = []

best_cnn_accuracy = []

best_cnn_epochs = []


for config in configs:


    config_result = results_df[
        results_df["Config ID"] == config
    ]


    config_best_index = config_result[
        "Test Accuracy"
    ].idxmax()


    config_best = results_df.loc[
        config_best_index
    ]


    configuration_name = config_names[
        config
    ]


    config_best_epoch = int(
        config_best["Epochs"]
    )


    config_best_accuracy = float(
        config_best["Test Accuracy (%)"]
    )


    print(

        f"{configuration_name:<40} | "

        f"Best Epoch = {config_best_epoch:3d} | "

        f"Accuracy = {config_best_accuracy:.2f}%"
    )


    best_cnn_names.append(
        f"CNN {config}"
    )


    best_cnn_accuracy.append(
        config_best_accuracy
    )


    best_cnn_epochs.append(
        config_best_epoch
    )


# ================================================================
# GRAPH 8
# BEST ACCURACY OF EACH CNN
# ================================================================

plt.figure(
    figsize=(9, 7)
)


bars = plt.bar(

    best_cnn_names,

    best_cnn_accuracy,

    color=[
        "steelblue",
        "orange",
        "seagreen"
    ]
)


plt.title(
    "Best Accuracy of Each CNN Configuration"
)


plt.xlabel(
    "CNN Configuration"
)


plt.ylabel(
    "Best Test Accuracy (%)"
)


plt.grid(
    axis="y",
    alpha=0.3
)


for bar, accuracy_value, epoch_value in zip(

    bars,

    best_cnn_accuracy,

    best_cnn_epochs
):

    plt.text(

        bar.get_x()
        +
        bar.get_width() / 2,

        bar.get_height()
        +
        0.2,

        (
            f"{accuracy_value:.2f}%\n"
            f"{epoch_value} Epochs"
        ),

        ha="center",

        va="bottom",

        fontsize=9
    )


plt.tight_layout()


graph8 = os.path.join(
    graph_folder,
    "08_best_cnn_accuracy.png"
)


plt.savefig(
    graph8,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ================================================================
# 27. SAVE CSV RESULTS
# ================================================================

results_file = os.path.join(
    current_folder,
    "cnn_results.csv"
)


predictions_file = os.path.join(
    current_folder,
    "cnn_predictions.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


prediction_df.to_csv(
    predictions_file,
    index=False
)


# ================================================================
# 28. SAVE BEST MODEL
# ================================================================

model_file = os.path.join(
    current_folder,
    "best_cnn_model.keras"
)


best_model.save(
    model_file
)


# ================================================================
# 29. ALL GRAPH FILES
# ================================================================

graph_files = [

    graph1,

    graph2,

    graph3,

    graph4,

    graph5,

    graph6,

    graph7,

    graph8
]


# ================================================================
# 30. CHECK ALL GRAPHS
# ================================================================

print("\n")
print("=" * 80)
print("CHECK SAVED GRAPHS")
print("=" * 80)


saved_graph_count = 0


for graph_path in graph_files:


    if os.path.exists(
        graph_path
    ):

        saved_graph_count += 1

        print(
            "[OK]",
            os.path.basename(
                graph_path
            )
        )


    else:

        print(
            "[ERROR]",
            os.path.basename(
                graph_path
            )
        )


print(
    f"\nSaved graphs: "
    f"{saved_graph_count}/8"
)


# ================================================================
# 31. FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)


print(
    f"""
Dataset
-------
WineQT.csv


Target
------
quality


CNN Type
--------
1D CNN


Training Data
-------------
80%


Testing Data
------------
20%


Standardization
---------------
StandardScaler


Epochs Tested
-------------
20
50
100


CNN Configurations
------------------

CNN 1
Conv1D 32
Dense 32


CNN 2
Conv1D 32
Conv1D 64
Dense 64


CNN 3
Conv1D 64
Conv1D 128
Dense 128
Dense 64


Total Experiments
-----------------
9


Best Configuration
------------------
{config_names[best_config]}


Best Epoch
----------
{best_epoch}


Best Test Accuracy
------------------
{best_accuracy:.2f}%


Best Test Loss
--------------
{best_loss:.4f}


Correct Predictions
-------------------
{correct_predictions}


Wrong Predictions
-----------------
{wrong_predictions}


Prediction Accuracy
-------------------
{final_accuracy * 100:.2f}%
"""
)


# ================================================================
# 32. FILE LOCATIONS
# ================================================================

print("=" * 80)
print("FILES SAVED")
print("=" * 80)


print(
    "\nResults CSV:"
)

print(
    results_file
)


print(
    "\nPredictions CSV:"
)

print(
    predictions_file
)


print(
    "\nBest Model:"
)

print(
    model_file
)


print(
    "\nGraph Folder:"
)

print(
    graph_folder
)


print("\nGraphs:")


for graph_path in graph_files:

    print(
        os.path.basename(
            graph_path
        )
    )


# ================================================================
# 33. PROGRAM FINISHED
# ================================================================

print("\n")
print("=" * 80)
print("PROGRAM FINISHED SUCCESSFULLY")
print("=" * 80)


print(
    "\nAll 8 graphs have been created."
)

print(
    "Opening all graph images..."
)


# ================================================================
# 34. OPEN ALL GRAPHS AT THE END
# ================================================================
#
# สำคัญ:
#
# ไม่มี plt.show() อยู่ระหว่างการทำงาน
#
# Program จะ:
#
# 1. Train
# 2. Test
# 3. Prediction
# 4. สร้างกราฟ 8 รูป
# 5. Save รูป
# 6. เปิดทั้ง 8 รูปตอนท้าย
#
# ================================================================

for graph_path in graph_files:

    if os.path.exists(
        graph_path
    ):

        try:

            os.startfile(
                graph_path
            )

        except Exception as error:

            print(
                "Cannot open:",
                graph_path
            )

            print(
                error
            )


print("\n")
print("=" * 80)
print("ALL DONE")
print("=" * 80)


input(
    "\nPress Enter to close program..."
)
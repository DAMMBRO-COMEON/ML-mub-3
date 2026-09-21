# LAB: CNN on a Dataset of Your Choice

## Wine Quality Classification using 1D CNN

This project applies a **1D Convolutional Neural Network (CNN)** to classify wine quality using the `WineQT.csv` dataset.

The project compares different:

- CNN configurations
- Numbers of convolutional layers
- Numbers of filters and neurons
- Numbers of training epochs
- Training, validation, and testing accuracy
- Training and validation loss

The model also performs predictions on the test dataset and generates graphs for performance analysis.

---

# 1. Objective

The objectives of this LAB are:

- Apply a CNN to classify a selected dataset.
- Split the dataset into training and testing sets.
- Standardize the input features before training.
- Build different CNN models.
- Train CNN models using different numbers of epochs.
- Compare different CNN configurations.
- Evaluate each model using accuracy.
- Compare training and validation accuracy.
- Compare training and validation loss.
- Perform predictions on the test dataset.
- Visualize model performance using graphs.

---

# 2. Dataset

Dataset used:

`WineQT.csv`

The dataset contains physicochemical properties of wine.

The target variable is:

`quality`

The `Id` column is not used as an input feature because it is only an identifier for each sample.

## Input Features

The model uses the following 11 features:

1. fixed acidity
2. volatile acidity
3. citric acid
4. residual sugar
5. chlorides
6. free sulfur dioxide
7. total sulfur dioxide
8. density
9. pH
10. sulphates
11. alcohol

## Target

The target column is:

`quality`

Possible wine quality classes in this dataset include:

- Quality 3
- Quality 4
- Quality 5
- Quality 6
- Quality 7
- Quality 8

---

# 3. Data Preprocessing

Before training the CNN, the dataset is preprocessed.

## 3.1 Remove ID

The `Id` column is removed because it does not represent a wine characteristic.

The following columns are therefore not used as input features:

- `quality` because it is the target
- `Id` because it is an identifier

---

## 3.2 Label Encoding

Wine quality values are converted into class numbers.

Example:

- Quality 3 -> Class 0
- Quality 4 -> Class 1
- Quality 5 -> Class 2
- Quality 6 -> Class 3
- Quality 7 -> Class 4
- Quality 8 -> Class 5

`LabelEncoder` from Scikit-learn is used for this process.

---

## 3.3 Train / Test Split

The dataset is divided into:

- Training Data: 80%
- Testing Data: 20%

The program uses:

`random_state = 42`

and stratified sampling to maintain a similar class distribution between the training and testing datasets.

---

## 3.4 Standardization

The input features have different scales.

For example, pH, alcohol, sulfur dioxide, and acidity have different numerical ranges.

Therefore, `StandardScaler` is used to standardize the input features.

The standardization formula is:

z = (x - mean) / standard deviation

The scaler is fitted only on the training dataset and then applied to the testing dataset.

---

# 4. Why 1D CNN?

The WineQT dataset contains tabular numerical features rather than images.

Therefore, this project uses:

`Conv1D`

instead of:

`Conv2D`

The original input shape is:

(samples, 11)

The data is reshaped into:

(samples, 11, 1)

This allows the numerical features to be processed using a one-dimensional convolutional neural network.

---

# 5. CNN Configurations

Three CNN configurations are compared.

## CNN Configuration 1

Architecture:

Input

Conv1D
- Filters: 32
- Kernel Size: 3
- Activation: ReLU

MaxPooling1D

Flatten

Dense
- Neurons: 32
- Activation: ReLU

Dropout
- Rate: 0.30

Output
- Activation: Softmax

Simplified structure:

Input
|
Conv1D(32)
|
MaxPooling1D
|
Flatten
|
Dense(32)
|
Dropout
|
Output

---

## CNN Configuration 2

Architecture:

Input

Conv1D
- Filters: 32
- Kernel Size: 3
- Activation: ReLU

Conv1D
- Filters: 64
- Kernel Size: 3
- Activation: ReLU

MaxPooling1D

Flatten

Dense
- Neurons: 64
- Activation: ReLU

Dropout
- Rate: 0.30

Output
- Activation: Softmax

Simplified structure:

Input
|
Conv1D(32)
|
Conv1D(64)
|
MaxPooling1D
|
Flatten
|
Dense(64)
|
Dropout
|
Output

---

## CNN Configuration 3

Architecture:

Input

Conv1D
- Filters: 64
- Kernel Size: 3
- Activation: ReLU

Conv1D
- Filters: 128
- Kernel Size: 3
- Activation: ReLU

MaxPooling1D

Flatten

Dense
- Neurons: 128
- Activation: ReLU

Dropout
- Rate: 0.30

Dense
- Neurons: 64
- Activation: ReLU

Dropout
- Rate: 0.20

Output
- Activation: Softmax

Simplified structure:

Input
|
Conv1D(64)
|
Conv1D(128)
|
MaxPooling1D
|
Flatten
|
Dense(128)
|
Dropout
|
Dense(64)
|
Dropout
|
Output

---

# 6. Epoch Comparison

Each CNN configuration is trained using three different numbers of epochs:

- 20 Epochs
- 50 Epochs
- 100 Epochs

Therefore, the project performs a total of:

3 CNN Configurations x 3 Epoch Settings = 9 Experiments

The experiments are:

1. CNN 1 - 20 Epochs
2. CNN 1 - 50 Epochs
3. CNN 1 - 100 Epochs
4. CNN 2 - 20 Epochs
5. CNN 2 - 50 Epochs
6. CNN 2 - 100 Epochs
7. CNN 3 - 20 Epochs
8. CNN 3 - 50 Epochs
9. CNN 3 - 100 Epochs

---

# 7. Model Training

The CNN models use the following training configuration:

Optimizer:

`Adam`

Learning Rate:

`0.001`

Loss Function:

`categorical_crossentropy`

Evaluation Metric:

`accuracy`

Batch Size:

`32`

Validation Data:

20% of the training dataset

---

# 8. Model Evaluation

Each experiment is evaluated using:

- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Training Loss
- Validation Loss
- Test Loss

The program automatically compares all nine experiments.

The model with the highest test accuracy is selected as the best model.

---

# 9. Prediction

After selecting the best CNN configuration, the model performs predictions on the test dataset.

The prediction results contain:

- Actual Quality
- Predicted Quality
- Correct / Incorrect

Example:

Actual Quality: 6

Predicted Quality: 6

Correct: True

The final prediction accuracy is also calculated.

---

# 10. Confusion Matrix

A confusion matrix is generated to compare:

Actual Wine Quality

with

Predicted Wine Quality

This helps identify which wine quality classes are classified correctly and which classes are confused with other classes.

---

# 11. Output Graphs

The program automatically generates 8 graph images.

All graph files are stored inside:

`CNN_Graphs/`

---

## Graph 1: Wine Quality Distribution

File:

`01_quality_distribution.png`

Purpose:

Shows the number of samples for each wine quality class.

This graph is useful for checking whether the dataset is balanced or imbalanced.

---

## Graph 2: Training and Validation Accuracy

File:

`02_training_validation_accuracy.png`

Purpose:

Compares training accuracy and validation accuracy over the training epochs of the best model.

This graph shows how the classification performance changes during training.

---

## Graph 3: Training and Validation Loss

File:

`03_training_validation_loss.png`

Purpose:

Compares training loss and validation loss over the training epochs.

This graph can help identify whether the model is learning correctly or beginning to overfit.

---

## Graph 4: Epoch vs Accuracy

File:

`04_epoch_vs_accuracy.png`

Purpose:

Compares test accuracy at:

- 20 Epochs
- 50 Epochs
- 100 Epochs

for all three CNN configurations.

This graph directly shows the effect of the number of epochs on CNN performance.

---

## Graph 5: CNN Configuration Comparison

File:

`05_cnn_configuration_comparison.png`

Purpose:

Compares the accuracy of:

- CNN 1
- CNN 2
- CNN 3

using 20, 50, and 100 epochs.

This graph helps determine which CNN architecture performs best.

---

## Graph 6: Actual vs Predicted

File:

`06_actual_vs_predicted.png`

Purpose:

Compares the actual wine quality values with the values predicted by the best CNN model.

This provides a visual representation of prediction performance.

---

## Graph 7: Confusion Matrix

File:

`07_confusion_matrix.png`

Purpose:

Shows the number of correct and incorrect predictions for each wine quality class.

The diagonal values represent correct predictions.

---

## Graph 8: Best CNN Accuracy

File:

`08_best_cnn_accuracy.png`

Purpose:

Shows the best test accuracy obtained by each CNN configuration.

It also displays the number of epochs that produced the best result for each CNN model.

---

# 12. Generated Files

After the program finishes, the project directory will contain:

CNN.py

WineQT.csv

cnn_results.csv

cnn_predictions.csv

best_cnn_model.keras

CNN_Graphs/

The `CNN_Graphs` directory contains:

01_quality_distribution.png

02_training_validation_accuracy.png

03_training_validation_loss.png

04_epoch_vs_accuracy.png

05_cnn_configuration_comparison.png

06_actual_vs_predicted.png

07_confusion_matrix.png

08_best_cnn_accuracy.png

---

# 13. CSV Output Files

## cnn_results.csv

Contains the results from all CNN experiments.

Important columns include:

- Configuration
- Epochs
- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Training Loss
- Validation Loss
- Test Loss

---

## cnn_predictions.csv

Contains the prediction results from the best model.

Important columns include:

- Actual Quality
- Predicted Quality
- Correct

---

# 14. Saved Model

The best-performing CNN model is automatically saved as:

`best_cnn_model.keras`

This model can be loaded again without training all nine experiments.

---

# 15. Project Structure

The recommended project structure is:

ML-07/
|
|-- CNN.py
|-- WineQT.csv
|-- README.md
|
|-- cnn_results.csv
|-- cnn_predictions.csv
|-- best_cnn_model.keras
|
|-- CNN_Graphs/
    |
    |-- 01_quality_distribution.png
    |-- 02_training_validation_accuracy.png
    |-- 03_training_validation_loss.png
    |-- 04_epoch_vs_accuracy.png
    |-- 05_cnn_configuration_comparison.png
    |-- 06_actual_vs_predicted.png
    |-- 07_confusion_matrix.png
    |-- 08_best_cnn_accuracy.png

---

# 16. Requirements

The project requires Python and the following libraries:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow

Install the required libraries using:

pip install numpy pandas matplotlib scikit-learn tensorflow

---

# 17. How to Run

Make sure the following files are in the same folder:

`CNN.py`

`WineQT.csv`

Then run:

python CNN.py

For example, on Windows:

C:\Users\SW02\AppData\Local\Programs\Python\Python313\python.exe CNN.py

The program will automatically:

1. Load the WineQT dataset
2. Check missing values
3. Prepare input features and target
4. Encode the target classes
5. Split training and testing data
6. Standardize the input features
7. Reshape the data for Conv1D
8. Create three CNN configurations
9. Train each model using 20, 50, and 100 epochs
10. Evaluate all nine experiments
11. Find the best CNN model
12. Perform predictions
13. Calculate prediction accuracy
14. Generate a confusion matrix
15. Save all eight graphs
16. Save experiment results
17. Save prediction results
18. Save the best CNN model

---

# 18. Expected Output

During execution, the program displays results similar to:

Experiment 1/9

Model  : CNN 1: Conv32 + Dense32
Epochs : 20

Training Accuracy   : xx.xx%
Validation Accuracy : xx.xx%
Test Accuracy       : xx.xx%

The accuracy values depend on the actual training results.

The program should not use manually created accuracy values. All reported accuracy and loss values are obtained from model training and evaluation.

---

# 19. Conclusion

This project demonstrates the application of a 1D Convolutional Neural Network to wine quality classification.

Three different CNN configurations are evaluated using three different numbers of training epochs.

A total of nine experiments are performed.

The experiments allow comparisons of:

- CNN architecture
- Number of convolutional layers
- Number of filters
- Number of neurons
- Number of epochs
- Training accuracy
- Validation accuracy
- Test accuracy
- Training loss
- Validation loss
- Test loss

The CNN configuration with the highest test accuracy is automatically selected as the best model.

The final model is then used to predict wine quality on the testing dataset.

Graphs and CSV files are automatically generated to provide clear evidence for the comparison and evaluation required by the LAB.
# Wine Quality Classification Using Artificial Neural Network (ANN)

## Project Overview

This project aims to classify wine quality levels using an Artificial Neural Network (ANN) implemented with TensorFlow/Keras. The dataset contains various physicochemical properties of wine, and the model predicts the corresponding quality score.

Several experiments are conducted to evaluate the impact of:

- Different numbers of training epochs
- Different neural network architectures
- Model performance on unseen test data

The project demonstrates a complete machine learning workflow, including data preprocessing, feature scaling, model training, evaluation, prediction, and performance visualization.

---

## Dataset

The project uses the **Wine Quality Dataset (WineQT.csv)**.

### Features

The input features include:

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

### Target

- **Quality** (Wine Quality Score)

The `Id` column is excluded because it serves only as an identifier and does not contribute to prediction performance.

---

## Project Workflow

### 1. Data Loading

The dataset is loaded using Pandas and inspected to understand:

- Dataset shape
- Data types
- Missing values
- Sample records

### 2. Data Preparation

The dataset is separated into:

- Features (`X`)
- Target labels (`y`)

The `quality` column is used as the target variable.

### 3. Label Encoding

Wine quality values are transformed into numerical classes using `LabelEncoder`.

Example:

| Original Quality | Encoded Class |
|------------------|--------------|
| 3 | 0 |
| 4 | 1 |
| 5 | 2 |
| 6 | 3 |
| 7 | 4 |
| 8 | 5 |

### 4. Train-Test Split

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

Stratified sampling is applied to preserve class distribution.

### 5. Feature Scaling

Standardization is performed using `StandardScaler`.

Benefits:

- Faster convergence
- Improved neural network performance
- Better numerical stability

### 6. Neural Network Construction

The ANN model is implemented using TensorFlow/Keras.

Key characteristics:

- ReLU activation for hidden layers
- Softmax activation for output layer
- Adam optimizer
- Sparse Categorical Crossentropy loss function

---

## Experiment 1: Epoch Comparison

Different epoch values are tested to examine their impact on classification accuracy.

Tested Epoch Values:

- 20
- 50
- 100

For each experiment:

1. Train the network
2. Evaluate on the test dataset
3. Record classification accuracy

The results are summarized in a comparison table.

---

## Experiment 2: Neural Network Architecture Comparison

Several network architectures are evaluated.

### Configuration 1 (NN-1)

```python
[16]
```

- One hidden layer
- 16 neurons

### Configuration 2 (NN-2)

```python
[16, 8]
```

- Two hidden layers
- 16 neurons
- 8 neurons

### Configuration 3 (NN-3)

```python
[64, 32, 16]
```

- Three hidden layers
- 64 neurons
- 32 neurons
- 16 neurons

Each model is trained for 50 epochs and evaluated using test accuracy.

---

## Final Model

The selected architecture is:

```python
[16, 8]
```

Training configuration:

- Epochs: 50
- Batch Size: 32
- Validation Split: 20%
- Optimizer: Adam
- Activation Function: ReLU
- Output Activation: Softmax

---

## Evaluation Metrics

The model performance is measured using:

### Test Loss

Measures prediction error on unseen data.

### Test Accuracy

Measures the percentage of correctly classified samples.

### Sklearn Accuracy Score

A secondary accuracy calculation performed using:

```python
accuracy_score()
```

to validate prediction performance.

---

## Prediction Process

The trained model produces probability distributions for each quality class.

The predicted class is selected using:

```python
np.argmax()
```

Then the encoded classes are converted back to their original wine quality values using:

```python
encoder.inverse_transform()
```

Example output:

```text
Predicted:
[6 5 6 7 5 ...]

Actual:
[6 5 5 7 5 ...]
```

---

## Visualization

### Training vs Validation Accuracy

This graph shows:

- Training Accuracy
- Validation Accuracy

It helps determine whether the model is learning effectively and if overfitting occurs.

### Training vs Validation Loss

This graph shows:

- Training Loss
- Validation Loss

It helps evaluate convergence behavior during training.

---

## Libraries Used

```python
pandas
numpy
matplotlib
scikit-learn
tensorflow
keras
```

Install required packages:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

---

## Running the Project

1. Place the dataset file (`WineQT.csv`) in the desired location.
2. Update the dataset path in the source code if necessary.
3. Run the script:

```bash
python wine_quality_ann.py
```

---

## Expected Outputs

The program will display:

- Dataset information
- Epoch comparison results
- Neural network configuration comparison
- Final testing accuracy
- Sample predictions
- Accuracy score
- Accuracy graph
- Loss graph

---

## Conclusion

This project demonstrates the application of Artificial Neural Networks for multiclass classification of wine quality. By comparing multiple epoch values and network architectures, the project identifies an effective ANN configuration and evaluates its performance using unseen test data. The complete workflow provides practical experience with data preprocessing, neural network design, model evaluation, and visualization in Python using TensorFlow and Scikit-learn.

---

## Author

**Veerapat Wandee**

Computer Engineering Student

Machine Learning Laboratory Project
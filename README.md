# Iris-KNN-Classifier

This project implements a **k-Nearest Neighbors (k-NN) classifier** on the classic **Iris dataset** using Python and scikit-learn.

## Features
- Train/test split with reproducible results (`random_state=1`)
- Model training and evaluation
- Accuracy, confusion matrix, and classification report
- Displays actual vs predicted class names

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/Salem9809/Iris-KNN-Classifier.git
cd Iris-KNN-Classifier

Model Accuracy: 97.78%

Confusion Matrix:
[[14  0  0]
 [ 0 18  0]
 [ 0  1 12]] 

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        14
  versicolor       0.95      1.00      0.97        18
   virginica       1.00      0.92      0.96        13

    accuracy                           0.98        45
   macro avg       0.98      0.97      0.98        45
weighted avg       0.98      0.98      0.98        45

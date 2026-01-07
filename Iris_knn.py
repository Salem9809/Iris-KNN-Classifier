"""
Iris Dataset Classifier using k-Nearest Neighbors (k-NN)

This script trains a k-NN classifier on the classic Iris dataset
and evaluates its accuracy on a test set.

Author: Salem Abdulla
Date: 7-1-2026
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

    # Split dataset into train and test sets (70/30 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.3, random_state = 1
    )

    # Initialize k-NN classifier
    knn = KNeighborsClassifier()
    
    # Train the model
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy*100:.2f}%\n")

    # Confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred), "\n")

    # Detailed classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Show actual vs predicted with class names
    print("Actual vs Predicted:")
    for actual, pred in zip(y_test, y_pred):
        print(f"{target_names[actual]} -> {target_names[pred]}")

if __name__ == "__main__":
    main()

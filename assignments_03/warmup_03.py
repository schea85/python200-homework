import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing ---

# Preprocessing Q1:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Preprocessing Q2:
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nX_train_scaled mean:", X_train_scaled.mean(axis=0))
print("X_test_scaled mean:", X_test_scaled.mean(axis=0))
# We fit the scaler on X_train to prevent data leakage from the test set.

# --- KNN ---

# KNN Q1:
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

preds_1 = knn.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, preds_1))
print("\nClassification Report:\n", classification_report(y_test, preds_1))

# KNN Q2:
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

preds_2 = knn.predict(X_test_scaled)

print("\nScaled Accuracy:", accuracy_score(y_test, preds_2))
print("\nScaled Classification Report:\n", classification_report(y_test, preds_2))
# Scaling generally improves or maintains performance for KNN.
# In this case, it hurt performance because the Iris features are already
# measured in centimeters and have similar ranges. Scaling reduced the natural separation
# that raw petal measurements provide between species.

# KNN Q3:
knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

print("\nCV score:", cv_scores)
print(f"CV Mean: {cv_scores.mean():.3f}")
print(f"CV Std: {cv_scores.std():.3f}\n")
# Yes, the result is more trustworthy than a single train/test split.
# The model trains on 4 folds and evaluate on the fifth, repeating this
# process 5 times with a different validation fold each time.
# Averaging the scores provides a more consistent estimate of model performance.

# KNN Q4:
k_values = list(range(1, 16, 2))

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k}: mean={scores.mean():.6f}")

# I would choose k=5 because it achieved the highest cross-validation accuracy.
# k=7 had the same score; however it has more neighbors.  
# So k=5 is a simpler model and which I would use.

# --- Classifier Evaluation ---

# Classifier Evaluation Q1:
cm = confusion_matrix(y_test, preds_1)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("KNN Confusion Matrix (Iris)")
plt.savefig("assignments_03/outputs/knn_confusion_matrix.png")
plt.show()
# This model did not confuse any species; all predictions were correct.

# --- The sklearn API: Decision Trees ---

# Decision Trees Q1:

# --- Logistic Regression and Regularization ---

# Logistic Regression Q1:

# --- PCA ---

# PCA Q1:
# PCA Q2:
# PCA Q3:
# PCA Q4:

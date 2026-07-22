import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- ROC and AUC ---
# ROC Q1:

# logistic regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

log_reg_probs = log_reg.predict_proba(X_test)[:, 1]
log_reg_auc = roc_auc_score(y_test, log_reg_probs)

print(f"Logistic Regression AUC: {log_reg_auc:.3f}")

# KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train_scaled, y_train)

knn_probs = clf.predict_proba(X_test_scaled)[:, 1]
knn_auc = roc_auc_score(y_test, knn_probs)

print(f"KNN AUC: {knn_auc:.3f}")

# Comment:
# KNN has a higher AUC (0.939), so it separates the two classes better than 
# Logistic Regression, independent of the classification threshold.

# ROC Q2:

fpr1, tpr1, thresholds1 = roc_curve(y_test, log_reg_probs)

fpr2, tpr2, thresholds2 = roc_curve(y_test, knn_probs)

fig, ax = plt.subplots(figsize=(6,5))
RocCurveDisplay(fpr=fpr1, tpr=tpr1).plot(ax=ax, name=f"Logistic Regression (AUC={log_reg_auc:.2f})")
RocCurveDisplay(fpr=fpr2, tpr=tpr2).plot(ax=ax, name=f"KNN (AUC={knn_auc:.2f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
ax.set_title("ROC Comparison")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
plt.tight_layout()
plt.savefig("assignments_04/outputs/roc_comparison.png")
plt.show()

# logistic regression threshold closest to target TPR of .80
target_tpr = .80

idx = np.argmin(np.abs(tpr1 - target_tpr))

print("Logistic Regression:")
print(f"TPR: {tpr1[idx]:.3f}")
print(f"FPR: {fpr1[idx]:.3f}")

# KNN threshold closest to target TPR of .80
idx2 = np.argmin(np.abs(tpr2 - target_tpr))

print("KNN:")
print(f"TPR: {tpr2[idx2]:.3f}")
print(f"FPR: {fpr2[idx2]:.3f}")

# Comment:
# At TPR = 0.80, KNN has a lower FPR.
# This means that if we need to catch about 80% of positive cases,
# KNN would produce fewer false alarms than Logistic Regression.

# ROC Q3:

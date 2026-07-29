import os
import numpy as np
import pandas as pd
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
    f1_score
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

print("\nLogistic Regression:")
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
f1_scores = []

for threshold in thresholds1:
    y_pred = (log_reg_probs >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)
    
best_idx = np.argmax(f1_scores)

best_threshold = thresholds1[best_idx]
best_f1 = f1_scores[best_idx]
best_tpr = tpr1[best_idx]
best_fpr = fpr1[best_idx]

print(f"\nBest threshold: {best_threshold:.3f}")
print(f"TPR: {best_tpr:.3f}")
print(f"FPR: {best_fpr:.3f}")
print(f"F1 Score: {best_f1:.3f}")

# Comment
# The optimal threshold (0.276) is lower than the default 0.5.
# A lower threshold helps catch more positives but increases false positives.
# This is useful when missing positives is more costly than false alarms.



# --- GridSearchCV ---

# GridSearch Q1:

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])

param_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc"
)

grid_search.fit(X_train, y_train)

print(f"\nBest C: {grid_search.best_params_['clf__C']}")
print(f"Best CV AUC: {grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_

y_probs_best = best_model.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_probs_best)

print(f"Test AUC: {test_auc:.3f}")

# Comment:
# No, the grid search did not pick the default C=1.0 as I would have guessed.
# Instead, it selected C=100.0.
# The test AUC was 0.706, showing no improvement from tuning C compared to the
# default setting.



# GridSearch Q2:
pipe2 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", DecisionTreeClassifier(random_state=42)),
])

tree_param_grid = {
    "clf__max_depth": [2, 3, 5, 8, None]
}

tree_grid_search = GridSearchCV(
    estimator=pipe2,
    param_grid=tree_param_grid,
    cv=5,
    scoring="roc_auc"
)

tree_grid_search.fit(X_train, y_train)

print(f"\nBest max_depth: {tree_grid_search.best_params_['clf__max_depth']}")
print(f"Best CV AUC: {tree_grid_search.best_score_:.3f}")

best_model2 = tree_grid_search.best_estimator_

y_probs_best2 = best_model2.predict_proba(X_test)[:, 1]

test_auc2 = roc_auc_score(y_test, y_probs_best2)

print(f"Test AUC: {test_auc2:.3f}")

# Comment:
# The Decision Tree had a higher AUC than Logistic Regression, so I would bring the 
# Decision Tree forward for further development.  However, AUC is not the only
# factor to consider; complexity, speed, and the cost of false
# positives and false negatives are also important.



# GridSearch Q3:
results1 = pd.DataFrame(grid_search.cv_results_)

print("\nMean and STD of CV AUC for each parameter value from best to worst:")
print(
    results1[["param_clf__C", "mean_test_score", "std_test_score"]]
    .sort_values("mean_test_score", ascending=False)
    .to_string(index=False)
)

# Comment:
# C=100.0 and C=10.0 have nearly identical mean AUCs.
# I would choose C=100.0 because it has a slightly lower STD.
# (lower STD = more consistent, more reliable)



# --- joblib ---
print("\n")

# joblib Q1:
best_lr_pipe = best_model

joblib.dump(best_lr_pipe, "assignments_04/models/warmup_model.pkl")

loaded_clf = joblib.load("assignments_04/models/warmup_model.pkl")

original_preds = best_lr_pipe.predict(X_test)
loaded_preds   = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")

# Comment:
# Saving only the Logistic Regression model would lose the scaler.
# The model would receive unscaled data during prediction, causing
# incorrect results.



# joblib Q2:

# --- Simulated prediction script ---
loaded_model = joblib.load("assignments_04/models/warmup_model.pkl")

# Three hand-crafted test cases — raw, unscaled data
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

predictions = loaded_model.predict(new_samples)
probabilities = loaded_model.predict_proba(new_samples)[:, 1]

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Row {i+1}: Class={pred}, Probability={prob:.2f}")
    
# Comment:
# The all-zeros row prediction depends on the model's learned intercept because
# the features are near the average after scaling.
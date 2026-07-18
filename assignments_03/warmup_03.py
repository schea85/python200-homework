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
clf = DecisionTreeClassifier(max_depth=3, random_state=42)

clf.fit(X_train, y_train)

tree_preds = clf.predict(X_test)

print("\nDecision Tree Accuracy:", accuracy_score(y_test, tree_preds))
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, tree_preds))

# The Decision Tree accuracy was comparable to KNN. Both models performed well on the 
# Iris dataset.
# Scaling would not affect the Decision Tree b/c Decision Tree do not rely on distance
# calculations.  They split data based on feature values, so standardized features
# are not necessary.

# --- Logistic Regression and Regularization ---

# Logistic Regression Q1:
log_reg_full_1 = LogisticRegression(C=0.01, max_iter=1000)
log_reg_full_1.fit(X_train_scaled, y_train)
print("\nC=0.01, coefficients=", np.abs(log_reg_full_1.coef_).sum())

log_reg_full_2 = LogisticRegression(C=1.0, max_iter=1000)
log_reg_full_2.fit(X_train_scaled, y_train)
print("C=1.0, coefficients=", np.abs(log_reg_full_2.coef_).sum())

log_reg_full_3 = LogisticRegression(C=100, max_iter=1000)
log_reg_full_3.fit(X_train_scaled, y_train)
print("C=100, coefficients=", np.abs(log_reg_full_3.coef_).sum())

# As C increases, the total coefficient magnitude increases b/c regularization becomes
# weaker.  Smaller C values shrink coefficients more strongly, making the model more stable.

# --- PCA ---
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# PCA Q1:
print("\nX_digits shape:", X_digits.shape)
print("images shape:", images.shape)

unique_digits = np.unique(y_digits)
num_digits = len(unique_digits)

fig, axes = plt.subplots(1, 10, figsize=(8, 4))
for ax, digit in zip(axes, unique_digits):
    idx = np.where(y_digits == digit)[0][0]
    
    # plot
    ax.imshow(images[idx], cmap="gray_r")
    ax.set_title(str(digit))
    ax.axis("off")
    
plt.tight_layout()

plt.savefig("assignments_03/outputs/sample_digits.png")
plt.show()

# PCA Q2:
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
plt.colorbar(scatter, label='Digit')
plt.title("PCA 2D Projection of Digits")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.savefig("assignments_03/outputs/pca_2d_projection.png")
plt.show()

# yes, same-digit images tend to cluster together in this 2D space; with 
# a little overlap here and there.

# PCA Q3:
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
x_values = np.arange(1, 65)

plt.plot(x_values, cumulative_variance)
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")

plt.savefig("assignments_03/outputs/pca_variance_explained.png")
plt.show()

# Approximately 13 principal components are needed to explain about 
# 80% of the variance.

# PCA Q4:
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_values = [2, 5, 15, 40]

fig, axes = plt.subplots(5, 5, figsize=(10, 10))

# original digits (top row)
for col in range(5):
    axes[0, col].imshow(images[col], cmap="gray_r")
    axes[0, col].axis("off")
    if col == 0:
        axes[0, col].set_ylabel("Original", fontsize=12)
    
# reconstructions
for row, n in enumerate(n_values, start=1):
    for col in range(5):
        reconstruction = reconstruct_digit(col, scores, pca, n)
        
        axes[row, col].imshow(reconstruction, cmap="gray_r")
        axes[row, col].axis("off")
        if col == 0:
            axes[row, col].set_ylabel(f"n={n}", fontsize=12)
        
plt.tight_layout()
plt.savefig("assignments_03/outputs/pca_reconstructions.png")
plt.show()

# Digits become clearly recognizable around n=15 components.
# This matches the variance curve because most important info is captured
# with the first several components.
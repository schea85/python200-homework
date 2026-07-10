import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# --- scikit-learn API ---
# scikit-learn Q1:
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
new_years = np.array([4, 8]).reshape(-1, 1)

model = LinearRegression()
model.fit(years, salary)
salary_predicted = model.predict(new_years)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted salaries:", salary_predicted)

# scikit-learn Q2:
x = np.array([10, 20, 30, 40, 50])
print("Original Shape:", x.shape)
x_2 = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
print("Reshaped:", x_2.shape)
# X must be 2D because each row represents one sample and each column represents one feature.

# scikit-learn Q3:
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
print("X clusters' shape:", X_clusters.shape)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)

print("KMeans cluster centers:\n", kmeans.cluster_centers_)
print("Points in each cluster:", np.bincount(labels))

plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap="viridis")
plt.title("Clusters Found by K-Means")
plt.xlabel("Synthetic Scale")

plt.show()

# --- Linear Regression ---
# Linear Regression Q1
# Linear Regression Q2
# Linear Regression Q3
# Linear Regression Q4
# Linear Regression Q5

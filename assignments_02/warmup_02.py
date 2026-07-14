import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

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
print("Predicted salary for 4 years experience:", round(salary_predicted[0], 2))
print("Predicted salary for 8 years experience:", round(salary_predicted[1], 2))

# scikit-learn Q2:
x = np.array([10, 20, 30, 40, 50])
print("Original Shape:", x.shape)
x_2 = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
print("Reshaped:", x_2.shape)
# Scikit-learn requires X to be 2D because it expect data organized as samples(rows) and features(columns).
# Even when using only 1 feature, the input must still have a column dimension so the model knows which
# variables are being used for predictions.

# scikit-learn Q3:
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
print("X clusters' shape:", X_clusters.shape)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)

print("KMeans cluster centers:\n", kmeans.cluster_centers_)
print("Points in each cluster:", np.bincount(labels))

plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap="viridis", s=6, alpha=0.7)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="black",
    marker="X",
    s=200
)

plt.title("Clusters Found by K-Means")
plt.xlabel("x")
plt.ylabel("y")

plt.savefig("assignments_02/outputs/kmeans_clusters.png")
plt.show()

# --- Linear Regression ---
np.random.seed(42)
num_patients = 100
age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Linear Regression Q1
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")

plt.savefig("assignments_02/outputs/cost_vs_age.png")
plt.show()
#   the plot shows two groups of points, suggesting that smoker status affects
#   medical cost, with smokers generally having higher costs.

# Linear Regression Q2
X = age.reshape(-1, 1)
y = cost

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X train(Age) shape:", X_train.shape)
print("X test(Age) shape:", X_test.shape)
print("y train(Cost) shape:", y_train.shape)
print("y test(Cost) shape:", y_test.shape)

# Linear Regression Q3
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)
print("RMSE:", rmse)
print("R²:", r2)
#   the slope means that for every additional year of age, predicted medical costs increase
#   by the slope amount.

# Linear Regression Q4
X_full = np.column_stack([age, smoker])
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)

print("R²:", model_full.score(X_test_f, y_test_f))
print("age coefficient:", model_full.coef_[0])
print("smoker coefficient:", model_full.coef_[1])
#   the smoker coefficient represents the estimated increase in medical costs for smokers
#   compared to non-smokers, while keeping age constants.

# Linear Regression Q5
y_pred_full = model_full.predict(X_test_f)

plt.scatter(y_pred_full, y_test_f)

# create diagonal reference line
min_value = min(y_pred_full.min(), y_test_f.min())
max_value = max(y_pred_full.max(), y_test_f.max())

plt.plot([min_value, max_value], [min_value, max_value])

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Medical Cost")
plt.ylabel("Actual Medical Cost")

plt.savefig("assignments_02/outputs/predicted_vs_actual.png")
plt.show()

#   Points above mean the actual cost was higher than the predicted cost, so the model
#   underestimated.  Points below mean the actual cost was lower than the predicted cost,
#   so the model overestimated.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import seaborn as sns

# Pre-preprocessing
#   ; semicolon used for separator 

# --- TASK 1 ---
# load data
df = pd.read_csv("assignments_02/student_performance_math.csv", sep=";")

print("\nDataset shape:", df.shape)
print("\nDataset first 5 rows:\n", df.head(5))
print("\nDataset columns datatypes:\n", df.dtypes)

# histogram
plt.hist(df["G3"], bins=21, edgecolor="black")
plt.title("Distribution of Final Math Grades")
plt.xlabel("Grades")
plt.ylabel("Frequency")

plt.savefig("assignments_02/outputs/g3_distribution.png")
plt.show()

# --- TASK 2 ---
df_clean = df.copy()
df_clean = df_clean[df_clean["G3"] != 0]
print("Dataset before filtering out G3=0:", df.shape)
print("Dataset after filtering out G3=0:", df_clean.shape)
#   G3 = 0 represents students who missed the final exam, not a true score of 0 on exam.
#   keeping these rows would allow the model to learn that absent students had extremely poor
#   performance, which will distort predictions.

# convert yes/no columns to binary 0/1 columns
df_clean[["schoolsup", "internet", "higher", "activities"]] = df_clean[["schoolsup", "internet", "higher", "activities"]].replace({"yes": 1, "no": 0}).astype(int)

# convert gender column to binary 0/1 column
df_clean["sex"] = df_clean["sex"].replace({"F": 0, "M": 1}).astype(int)

# pearson correlations
r1, p1 = pearsonr(df["absences"], df["G3"])
r2, p2 = pearsonr(df_clean["absences"], df_clean["G3"])
print(f"Pearson before filtering: r={r1:.4f}, p={p1:.4f}")
print(f"Pearson after filtering: r={r2:.4f}, p={p2:.4f}")
#   filtering out the G3 = 0 changes the correlation b/c those students missed the final exam.
#   Their 0 grades are due to absence, not poor academic performance, so including them weakens
#   the true relationship b/w absences and final grades.

# --- TASK 3 ---
# create loop to compute pearson
features = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "absences", "freetime",
    "goout", "Walc"
]
results = []

for feature in features:
    r, p = pearsonr(df_clean[f"{feature}"], df_clean["G3"])
    results.append((feature, round(r, 4), round(p, 4)))
    
results.sort(key = lambda x: x[1])

# sort pearson/r values
for feature, r, p in results:
    print(feature, r)
    
# visualization 1 - scatter
plt.scatter(df_clean["failures"], df_clean["G3"], color="green")
plt.title("Past Failures vs Final Grade (G3)")
plt.xlabel("Number of Past Failures")
plt.ylabel("Final Grade (G3)")

plt.savefig("assignments_02/outputs/failures_vs_g3_scatter.png")
plt.show()
#   This scatter plot looks different from the others I've seen.
#   This scatter plot shows a weak negative relationship between past failures and final grades.
#   Students with more past failures generally have lower Final Grade (G3).
#   Past failures is probably one of the many factors contributing to student's over Final Grade.

# visualization 2 - heatmap
plt.figure(figsize=(12, 8))

corr = df_clean.corr()

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")

plt.savefig("assignments_02/outputs/g3_heatmap.png")
plt.show()
#   overall, these are weak negative and positive correlations.
#   the r numbers are closer to 0.

# --- TASK 4 ---
X = df_clean[["failures"]]
y = df_clean["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)              

print("Slope:", round(model.coef_[0], 4))
print("RMSE:", round(rmse, 4))
print("R²:", round(r2, 4))
#   the slope shows that there is a negative correlation
#   the RMSE shows the average prediction error in grade points
#   the low R² suggests failures alone does not explain much of the variation in G3

# --- TASK 5 ---
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred = model_multi.predict(X_test_m)

r2_train = model_multi.score(X_train_m, y_train_m)         
print("Train R²:", round(r2_train, 4))
r2_test = model_multi.score(X_test_m, y_test_m)         
print("Test R²:", round(r2_test, 4))
rmse_2 = np.sqrt(np.mean((y_pred - y_test_m) ** 2))
print("Task 5 RMSE:", rmse_2)
print("\nCompare Task4 baseline to Task5 R² test:")
print(f"Task4 Test R²: {r2}")
print(f"Task5 Test R²: {r2_test}\n")
#   the full model has a higher R² than the baseline model,
#   showing that adding more features improves prediction accuracy.

for name, coef in zip(feature_cols, model_multi.coef_):
    print(f"{name:12s}: {coef:+.3f}")
#   schoolsup has a negative coefficient, which is surprising.  Possibly, because
#   if students are getting extra school support they were already struggling academically.

#   I would keep features like failures, studytime, higher, and internet because they had larger
#   coefficients, and consider dropping activities and freetime because they had
#   little impact on predictions.

# --- TASK 6 ----
plt.figure(figsize=(8, 6))

plt.scatter(y_pred, y_test_m)
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")

min_value = min(y_pred.min(), y_test_m.min())
max_value = max(y_pred.max(), y_test_m.max())

plt.plot([min_value, max_value],
         [min_value, max_value],
         color="red")

plt.savefig("assignments_02/outputs/predicted_vs_actual.png")
plt.show()

#   The filtered dataset has 357 rows and the test set has 72 rows. The RMSE of
#   2.855 means the model is usually off by about 2.9 points when predicting grades
#   on a 0-20 scale. The test R² of 0.154 means the model explains about 15.4% of the variation in
#   final grades. Internet (+0.834) and higher (+0.610) had the biggest positive coefficients, meaning
#   they are associated with higher predicted grades.
#   Schoolsup (-2.062) and failures (-1.145) had the largest negative coefficients,
#   meaning that they are associated with lower predicted grades. The negative schoolsup
#   result was surprising, but students receiving extra support may already be
#   struggling. Points above the diagonal mean the model predicted too low, while
#   points below mean it predicted too high.

# --- Neglected Feature: The Power of G1 ---
feature_cols_g1 = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                   "internet", "sex", "freetime", "activities", "traveltime", "G1"]
X_g1 = df_clean[feature_cols_g1].values
y_g1 = df_clean["G3"].values

X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1, y_g1, test_size=0.2, random_state=42
)

model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)

r2_g1 = model_g1.score(X_test_g1, y_test_g1)
print("Test R² with G1:", r2_g1)

#   Adding G1 greatly improves the R², but it does not mean G1 causes G3.  This model
#   is not useful for early intervention because G1 is already available.  To help
#   students earlier, educators would need to use other features before G1.


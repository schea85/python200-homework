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
    "goout", "Walc", "schoolsup", "internet", "higher", "activities", "sex"
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

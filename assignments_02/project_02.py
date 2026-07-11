import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

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
df_clean[["schoolsup", "internet", "higher", "activities"]] = df_clean[["schoolsup", "internet", "higher", "activities"]].replace({"yes": 1, "no": 0})

# convert gender column to binary 0/1 column
df_clean["sex"] = df_clean["sex"].replace({"F": 0, "M": 1})

# pearson correlations
r1, p1 = pearsonr(df["absences"], df["G3"])
r2, p2 = pearsonr(df_clean["absences"], df_clean["G3"])
print(f"Pearson before filtering: r={r1:.4f}, p={p1:.4f}")
print(f"Pearson after filtering: r={r2:.4f}, p={p2:.4f}")
#   filtering out the G3 = 0 changes the correlation b/c those students missed the final exam.
#   Their 0 grades are due to absence, not poor academic performance, so including them weakens
#   the true relationship b/w absences and final grades.

# --- TASK 3 ---


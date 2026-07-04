import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

# --- Pandas Review ---

# Pandas Q1:
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print(f"First 3 rows:\n", df.head(3))
print(f"\nShape:\n", df.shape)
print(f"\nData types:\n", df.dtypes)

# Pandas Q2:
filtered_grades = df[(df["passed"] == True) & (df["grade"] > 80)]
print(f"\nFiltered Grades over 80:\n", filtered_grades)

# Pandas Q3:
df["grade_curved"] = df["grade"] + 5
print(f"\nCurved grades:\n", df)

# Pandas Q4:
df["name_upper"] = df["name"].apply(str.upper)
print(f"\nDF w/ Uppercase Name Column:\n", df.loc[:, ["name", "name_upper"]])

# Pandas Q5:
avg_by_city = df.groupby("city")["grade"].mean()
print("\nAverage Grade by City:\n", avg_by_city)

# Pandas Q6:
df["city"] = df["city"].replace("Austin", "Houston")
print("\nReplace Austin with Houston:\n", df.loc[:, ["name", "city"]])

# Pandas Q7:
desc_grades = df.sort_values("grade", ascending=False)
print(f"\nTop 3 Highest Grades:\n", desc_grades.head(3))



# --- Numpy Review ---

# NumPy Q1:
array_1d= np.array([10, 20, 30, 40, 50])
print(f"\nArray 1D shape:\n", array_1d.shape)
print(f"\nArray 1D dtype:\n", array_1d.dtype)
print(f"\nArray 1D ndim:\n", array_1d.ndim)

# NumPy Q2:
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(f"\nArray 2D Shape:\n", arr.shape)
print(f"\nArray 2D Size:\n", arr.size)

# NumPy Q3:
print(f"\nSlice Top-Left :\n", arr[0:2, 0:2])

# NumPy Q4:
arr_zeros = np.zeros((3, 4))
arr_ones = np.ones((2, 5))
print(f"\nZeros:\n", arr_zeros)
print(f"\nOnes:\n", arr_ones)

# NumPy Q5:
q5_arr = np.arange(0, 50, 5)
print(f"\nArray Q5:\n", q5_arr)
print(f"\nArray Q5 Shape:\n", q5_arr.shape)
print(f"\nArray Q5 Mean:\n", np.mean(q5_arr))
print(f"\nArray Q5 Sum:\n", np.sum(q5_arr))
print(f"\nArray Q5 STD:\n", np.std(q5_arr))

# NumPy Q6:
q6_arr = np.random.normal(0, 1, 200)
print(f"\nArray Q6 Mean:\n", np.mean(q6_arr))
print(f"\nArray Q6 STD:\n", np.std(q6_arr))



# --- Matplotlib Review ---

# Matplotlib Q1:
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y)

plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

# Matplotlib Q2:
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

plt.bar(subjects, scores, color="blue")
plt.title("Subject Scores")
plt.xlabel("Subject")
plt.ylabel("Scores")

plt.show()

# Matplotlib Q3:
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color="green")
plt.scatter(x2, y2, color="blue")
plt.title("Scatter Plot of Two Datasets")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()

# Matplotlib Q4:
fig, axes = plt.subplots(1, 2, figsize=(10,4))

# left side
axes[0].plot(x, y)
axes[0].set_title("Squares")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

# right side
axes[1].bar(subjects, scores)
axes[1].set_title("Subject Scores")
axes[1].set_xlabel("Subjects")
axes[1].set_ylabel("Scores")

# Adjust spacing
plt.tight_layout()

plt.show()



# --- Descriptive Statistics Review ---

# Descriptive Stats Q1:
data_2 = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print(f"\nStats Q1 Mean:\n", np.mean(data_2))
print(f"\nStats Q1 Median:\n", np.median(data_2))
print(f"\nStats Q1 Variance:\n", np.var(data_2))
print(f"\nStats Q1 STD:\n", np.std(data_2))

# Descriptive Stats Q2:
data_normal = np.random.normal(65, 10, 500)

plt.hist(data_normal, bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")

plt.show()

# Descriptive Stats Q3:
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Score")

plt.show()

# Descriptive Stats Q4:
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.boxplot([normal_data, skewed_data], tick_labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")

plt.show()

# observation
# exponential is more skewed
# median is a better measure of central tendency for exponential (less affected by skew and extreme values)
# mean for normal

# Descriptive Stats Q5:
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print(f"\nData1 mean:", np.mean(data1))
print(f"\nData1 median:", np.median(data1))
print(f"\nData1 mode:", stats.mode(data1))

print(f"\nData2 mean:", np.mean(data2))
print(f"\nData2 median:", np.median(data2))
print(f"\nData2 mode:", stats.mode(data2))

# Because of the value 150 in data2; the mean is pulled upward by the outlier.

# --- Hypothesis Testing Review ---

# Hypothesis Q1:
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_val = stats.ttest_ind(group_a, group_b)

print("\nt-statistic:", t_stat)
print("\np-value:", p_val)

# Hypothesis Q2:
alpha = 0.05

if p_val < alpha:
    print("The difference is statistically significant.\n")
else:
    print("No statistically significant difference detected.\n")
    
# Hypothesis Q3:
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat2, p_val2 = stats.ttest_rel(before, after)

print(f"t-statistic: {t_stat2:.3f}")
print(f"p-value: {p_val2:.6f}\n")

# Hypothesis Q4:
scores = [72, 68, 75, 70, 69, 74, 71, 73]

t_stat3, p_val3 = stats.ttest_1samp(scores, 70)

print(f"t-statistic: {t_stat3:.3f}")
print(f"p-value: {p_val3:.6f}\n")

# Hypothesis Q5:
t_stat4, p_val4 = stats.ttest_ind(group_a, group_b, alternative="less")

print(f"One tailed p-value: {p_val4:.6f}\n")

# Hypothesis Q6:
print("The results suggest that Group B scored significantly higher than Group A, and this difference is unlikely to be due to chance.")



# --- Correlation Review ---

# Correlation Q1:
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)
print("Correlation matrix:", corr_matrix)
# correlation coefficient (x vs y)
corr_value = corr_matrix[0, 1]
print("\nCorrelation coefficient:", corr_value)
# Expected correlation to be 1 because y is a perfect linear transformation of x
# Positive correlation; as x increases so does y

# Correlation Q2:
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

corr, p_value = pearsonr(x, y)

print("\nCorrelation coefficient:", corr)
print("P-value:", p_value)

# Correlation Q3:
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
corr2 = df.corr()
print("Correlation matrix:\n", corr2)

# Correlation Q4:
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.scatter(x, y, color="orange")
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

# Correlation Q5:
sns.heatmap(corr2, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")

plt.show()



# --- Pipeline Q1 ---
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# step 1
def create_series(arr):
    values = pd.Series(arr, name="values")
    return values

# step 2
def clean_data(series):
    df_cleaned =  series.dropna()
    return df_cleaned

# step 3
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    
# pipeline
def data_pipeline(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary

# run pipeline
result = data_pipeline(arr)

# print
print("\nPipeline Q1:\n")
for key, value in result.items():
    print(f"{key}: {value}")
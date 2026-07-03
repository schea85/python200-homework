import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
# Matplotlib Q3:
# Matplotlib Q4:



# --- Descriptive Statistics Review ---

# Descriptive Stats Q1:
# Descriptive Stats Q2:
# Descriptive Stats Q3:
# Descriptive Stats Q4:
# Descriptive Stats Q5:



# --- Hypothesis Testing Review ---

# Hypothesis Q1:
# Hypothesis Q2:
# Hypothesis Q3:
# Hypothesis Q4:
# Hypothesis Q5:
# Hypothesis Q6:



# --- Correlation Review ---

# Correlation Q1:
# Correlation Q2:
# Correlation Q3:
# Correlation Q4:
# Correlation Q5:


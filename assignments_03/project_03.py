import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- TASK 1 ---

COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

print("\n", df["spam_label"].value_counts())

spam = df[df["spam_label"] == 1]
ham = df[df["spam_label"] == 0]

# boxplot 1
plt.figure(figsize=(8, 6))
plt.boxplot([spam["word_freq_free"], ham["word_freq_free"]], tick_labels = ["Ham", "Spam"])
plt.title("Spam vs. Ham: word_freq_free")
plt.xlabel("Email Type")
plt.ylabel("Word_Freq: 'free'")

plt.savefig("assignments_03/outputs/spam_vs_ham_word_freq_free.png")
plt.show()

# boxplot 2
plt.figure(figsize=(8, 6))
plt.boxplot([spam["char_freq_!"], ham["char_freq_!"]], tick_labels = ["Ham", "Spam"])
plt.title("Spam vs. Ham: char_freq_!")
plt.xlabel("Email Type")
plt.ylabel("Char_Freq: '!'")

plt.savefig("assignments_03/outputs/spam_vs_ham_char_freq_!.png")
plt.show()

# boxplot 3
plt.figure(figsize=(8, 6))
plt.boxplot([spam["capital_run_length_total"], ham["capital_run_length_total"]], tick_labels = ["Ham", "Spam"])
plt.title("Spam vs. Ham: capital_run_length_total")
plt.xlabel("Email Type")
plt.ylabel("capital_run_length_total")

plt.savefig("assignments_03/outputs/spam_vs_ham_capital_run_length_total.png")
plt.show()

# --- TASK 2 ---
X = df.drop("spam_label", axis=1)
y = df["spam_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA()
pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
x_values = np.arange(1, 58)

plt.plot(x_values, cumulative_variance)
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")

plt.savefig("assignments_03/outputs/project3_task2.png")
plt.show()

n = 0

for i, variance in enumerate(cumulative_variance):
    if variance >= 0.90:
        n = i + 1
        break
print("\nNumber of components for 90% variance:", n)

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]

# --- TASK 3 ---

# === KNeighborsClassifier - unscaled data ===
knn1 = KNeighborsClassifier(n_neighbors=5)
knn1.fit(X_train, y_train)

preds1 = knn1.predict(X_test)

print("\nKNN(unscaled) Accuracy:", accuracy_score(y_test, preds1))
print("KNN(unscaled) Classification Report:\n", classification_report(y_test, preds1))

# === KNeighborsClassifier - scaled data ===
scaler_2 = StandardScaler()
X_train_scaled = scaler_2.fit_transform(X_train)
X_test_scaled = scaler_2.transform(X_test)

knn2 = KNeighborsClassifier(n_neighbors=5)
knn2.fit(X_train_scaled, y_train)

preds2 = knn2.predict(X_test_scaled)

print("\nKNN(scaled) Accuracy:", accuracy_score(y_test, preds2))
print("KNN(scaled) Classification Report:\n", classification_report(y_test, preds2))

# === KNeighborsClassifier - PCA reduced data ===
knn3 = KNeighborsClassifier(n_neighbors=5)
knn3.fit(X_train_pca, y_train)

preds3 = knn3.predict(X_test_pca)

print("\nKNN - PCA-reduced data Accuracy:", accuracy_score(y_test, preds3))
print("KNN - PCA-reduced data  Classification Report:\n", classification_report(y_test, preds3))

# PCA did not improve accuracy (slightly worse) nor the classification report.

# === DecisionTreeClassifier ===
max_depths = [3, 5, 10, None]

for depth in max_depths:
    dtc = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dtc.fit(X_train, y_train)

    train_preds = dtc.predict(X_train)
    test_preds = dtc.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"\nMax Depth: {depth}")
    print(f"Training Accuracy: {train_acc}")
    print(f"Test Accuracy: {test_acc}")

# As max_depth increases, training accuracy increases and eventually reaches nearly 100%, indicating
# overfitting.  However, the unlimited tree achieved the highest test accuracy on this dataset,
# so I would choose max_depth=None.

dtc_final = DecisionTreeClassifier(max_depth=None, random_state=42)
dtc_final.fit(X_train, y_train)

final_preds = dtc_final.predict(X_test)

print("\nFinal Decision Tree Accuracy:", accuracy_score(y_test, final_preds))
print("\nFinal Decision Tree Classification Report:\n")
print(classification_report(y_test, final_preds))

# === RandomForestClassifier ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("\nRandom Forest Classification Report:\n")
print(classification_report(y_test, rf_preds))

# DecisionTree 10 Most Important Features
dtc_df = pd.DataFrame({
    "feature": X.columns,
    "importance": dtc_final.feature_importances_
})

dtc_df = dtc_df.sort_values(
    by="importance",
    ascending=False
)

# top 10
print("Decision Tree - Top 10 Most Important Features:")
print(dtc_df.head(10))


# RandomForest 10 Most Important Features
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

# top 10
top10_rf = importance_df.head(10)
print("\nRandom Forest - Top 10 Most Important Features:")
print(top10_rf)

# RandomForest bar chart
plt.figure(figsize=(10, 6))
plt.bar(top10_rf["feature"], top10_rf["importance"], color="skyblue", edgecolor="black")

plt.title("Random Forest - Top 10 Most Important Features")
plt.xlabel("Features")
plt.ylabel("Importance Value")
plt.xticks(rotation=45, ha="right")

plt.savefig("assignments_03/outputs/feature_importances.png")
plt.show()

# === LogisticRegression ===

# Logistic Regression - scaled
log_reg = LogisticRegression(
    C=1.0, 
    max_iter=1000,
    solver="liblinear"
)

log_reg.fit(X_train_scaled, y_train)

log_preds = log_reg.predict(X_test_scaled)

print("\nLogistic Regression Scaled Accuracy:", accuracy_score(y_test, log_preds))
print("Logistic Regression Scaled Classification Report:")
print(classification_report(y_test, log_preds))

# Logistic Regression - PCA-reduced data

log_reg_2 = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear"
)

log_reg_2.fit(X_train_pca, y_train)

log_preds_2 = log_reg_2.predict(X_test_pca)

print("\nLogistic Regression PCA-reduced Accuracy:", accuracy_score(y_test, log_preds_2))
print("Logistic Regression PCA-reduced Classification Report:")
print(classification_report(y_test, log_preds_2))

# Task 3 Summary:
# The best performing classifier was the Random Forest Model
# because it achieved the highest accuracy of 94.6%.
# PCA comparison: Non-PCA scaled models performed better; PCA did not help.
# Spam metric: Accuracy isn't enough; prioritize reducing false positives, but
# Random Forest had more false negatives than false positives.

# Random Forest - Confusion Matrix
cm = confusion_matrix(y_test, rf_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot()

plt.title("Random Forest Confusion Matrix")

plt.savefig("assignments_03/outputs/best_model_confusion_matrix.png")
plt.show()
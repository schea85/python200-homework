import requests
import pandas as pd
import joblib
import json
import sklearn
import sys
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, RocCurveDisplay

# Fetch Data
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 38.90,
    "longitude": -77.04,
    "start_date": "2020-01-01",
    "end_date": "2020-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max"
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("Weather Dataset Info:\n", df.info())
print("\nWeather Dataset Describe:\n", df.describe())
print("\nWeather Dataset - First 5 Rows:\n", df.head())
print("\nWeather Dataset Shape:", df.shape)

# Engineer Labels
def label_running_day(row):
    return int(
        10 <= row["temperature_2m_max"] <= 26
        and row["temperature_2m_min"] >= 4
        and row["precipitation_sum"] < 3.0 
        and row["wind_speed_10m_max"] < 25
    )
    
df["good_for_running"] = df.apply(label_running_day, axis=1)

print("\nClass Distribution:")
print(df["good_for_running"].value_counts())

fraction = df["good_for_running"].mean()
print(f"\nFraction of good running days: {fraction:.2%}")

# Comment:
# For the engineer labels, I most kept it similar to the lessons with minor tweaks.
# 87 good running days, or about 23.77%.  Which makes sense; I live in the DC area.
# We experience all 4 seasons, and summer and winter are too hot/cold to be outside.

# Train and Tune
FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max"
]

X = df[FEATURES]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_pipe = grid_search.best_estimator_
y_pred = best_pipe.predict(X_test)
y_probs = best_pipe.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_probs)

print("\nSummary:")
print(f"Best C: {grid_search.best_params_['clf__C']}")
print(f"Best CV AUC: {grid_search.best_score_:.3f}")
print("Classification Report:")
print(report)
print(f"Test AUC: {test_auc:.3f}")

# plot
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(
    fpr=fpr, 
    tpr=tpr,
    roc_auc=test_auc
).plot(ax=ax, name="Logistic Regression")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
ax.set_title("ROC Curve - Weather Classifier")
ax.legend()
plt.tight_layout()
plt.savefig("assignments_04/outputs/weather_roc.png")
plt.show()

# Reflection:
# The test AUC of 0.797 shows the model does a fairly good job of separating good and 
# bad running days, which is about what I expected using only four weather features.
# The classification report shows more false negatives because the recall for good running
# days is only 0.11, meaning that the model often misses days that are actually good for running.
# In practice, I would rather the app slightly over-recommend running than miss many good days.
# I would lower the decision threshold below 0.5 to increase recall, even if it creates
# more false positives.

# Save the model
joblib.dump(best_pipe, "assignments_04/models/weather_classifier.pkl")

metadata = {
    "python_version": sys.version,
    "sklearn_version": sklearn.__version__,
    "features": FEATURES,
    "label_threshold": {
        "temperature_2m_max": "7-26°C",
        "temperature_2m_min": ">= 4°C",
        "precipitation_sum": "< 3.0 mm",
        "wind_speed_10m_max": "< 25 km/h"
    },
    "best_params": grid_search.best_params_,
    "test_auc": round(test_auc, 4),
    "trained_on": "2020 Open-Mateo",
    "city": "Washington, DC (lat 38.90, lon -77.04)"
}

with open("assignments_04/models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
    
print("Model and metadata saved to models/")
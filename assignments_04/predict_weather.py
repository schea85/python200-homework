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

# load model
clf = joblib.load("assignments_04/models/weather_classifier.pkl")

# load metadata
with open("assignments_04/models/weather_classifier_metadata.json", "r") as f:
    metadata = json.load(f)
    
    print("Weather Metadata:")
    print("City:", metadata["city"])
    print("Features:", metadata["features"])
    print("Test AUC:", metadata["test_auc"])

new_days = pd.DataFrame({
    "temperature_2m_max": [20.0, 25.0, 2.0, 0.0, 27.0],
    "temperature_2m_min": [10.0, 2.0, 19.0, 0.0, 5.0],
    "precipitation_sum": [0.0, 0.0, 4.0, 6.0, 2.0],
    "wind_speed_10m_max": [18.0, 8.0, 35.0, 26.0, 25.0]
})

preds = clf.predict(new_days)
probs = clf.predict_proba(new_days)[:, 1]

for i, (pred, prob) in enumerate(zip(preds, probs)):
    label = "good for running" if pred == 1 else "skip"
    
    print(f"\nDay {i+1}")
    print(f"temperature_2m_max: {new_days.loc[i, 'temperature_2m_max']} °C")
    print(f"temperature_2m_min: {new_days.loc[i, 'temperature_2m_min']} °C")
    print(f"precipitation_sum: {new_days.loc[i, 'precipitation_sum']} mm")
    print(f"wind_speed_10m_max: {new_days.loc[i, 'wind_speed_10m_max']} km/h")
    print(f"Prediction: {label}")
    print(f"Probability of good for running: {prob:.2f}")
    
# Reflection:
# 1.) Day 2 was my borderline case with a probability of 0.53.
# The model's answer is uncertain because the probability is close to the 0.5 decision threshold.
# If the model predicted 0.52, I would treat the result with caution and consider other factors,
# like personal preference, before recommending a run.

# 2.) If predict_weather.py runs before train_weather_classifier.py, it will fail because the saved
# model and metadata files do not exist.  A better error message would tell the user
# to run the training script first.

# 3.) The predict_weather.py would need to fetch tomorrow's weather forecast automatically from API instead
# of using the one that was created.  It would then pass the forecast data to the trained model for prediction.
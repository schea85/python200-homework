import pandas as pd
import numpy as np
from prefect import task, flow

# --- TASK 1 ---

# loop
years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
happiness_dfs = []

for year in years:
    file = f"csv/world_happiness_{year}.csv"
    df = pd.read_csv(file, sep=";", decimal=",")
    df["year"] = year
    happiness_dfs.append(df)
    
# concat dfs
merged_happiness = pd.concat(happiness_dfs).reset_index(drop=True)

merged_happiness.to_csv("outputs/merged_happiness.csv")
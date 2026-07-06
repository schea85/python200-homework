import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prefect import flow, task
from prefect.runtime import task_run
from prefect.logging import get_run_logger
from random import random

# --- TASK 1 ---
@task(retries=3, retry_delay_seconds=2)
def load_data():
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

    merged_happiness.to_csv("assignments_01/outputs/merged_happiness.csv")
    
    return merged_happiness
    
# --- TASK 2 ---
@task
def get_stats(merged_happiness):
    logger = get_run_logger()
    
    happiness = merged_happiness.copy()
    
    # stats
    mean = np.mean(happiness["Happiness score"])
    median = np.median(happiness["Happiness score"])
    std = np.std(happiness["Happiness score"]) 
    
    logger.info(f"Mean happiness score: {mean:.3f}")
    logger.info(f"Median happiness score: {median:.3f}")
    logger.info(f"Standard deviation happiness score: {std:.3f}")
    
    # groupby region 
    happiness_mean_by_region = happiness.groupby("Regional indicator")["Happiness score"].mean()
    logger.info(f"Mean happiness by region:\n{happiness_mean_by_region}")
    
    # groupby year
    happiness_mean_by_year = happiness.groupby("year")["Happiness score"].mean()
    logger.info(f"Mean happiness by year:\n{happiness_mean_by_year}")
    
    return {
        "mean": mean,
        "median": median,
        "std": std
    }

# --- TASK 3: ---
@task
def create_visuals(merged_happiness):
    logger = get_run_logger()
    
    # histogram
    plt.hist(merged_happiness["Happiness score"], bins=10, color="purple", edgecolor="black")
    plt.title("Happiness Scores Histogram")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    # save histogram
    plt.savefig("assignments_01/outputs/happiness_histogram.png")
    logger.info("Histogram saved.")
    plt.close()
    
    # boxplot
    plt.figure()
    merged_happiness.boxplot(column="Happiness score", by="year")
    plt.title("Boxplot")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    # save boxplot
    plt.savefig("assignments_01/outputs/happiness_by_year.png")
    logger.info("Boxplot saved.")
    plt.close()
    
    # scatter plot
    plt.scatter(merged_happiness["GDP per capita"], merged_happiness["Happiness score"], color="green")
    plt.title("GDP vs Happiness score")
    plt.xlabel("GDP per capita")
    plt.ylabel("Happiness Score")
    # save scatter plot
    plt.savefig("assignments_01/outputs/gdp_vs_happiness.png")
    logger.info("Scatter plot saved.")
    plt.close()
    
    # heatmap
    plt.figure(figsize=(10,8))
    numeric_df = merged_happiness.select_dtypes(include="number") 
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    # save heatmap
    plt.savefig("assignments_01/outputs/correlation_heatmap.png")
    logger.info("Heatmap saved.")
    plt.close()
    
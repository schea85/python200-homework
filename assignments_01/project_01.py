import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
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

    merged_happiness.to_csv("outputs/merged_happiness.csv")
    
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
    plt.savefig("outputs/happiness_histogram.png")
    logger.info("Histogram saved.")
    plt.close()
    
    # boxplot
    plt.figure()
    merged_happiness.boxplot(column="Happiness score", by="year")
    plt.title("Boxplot")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    # save boxplot
    plt.savefig("outputs/happiness_by_year.png")
    logger.info("Boxplot saved.")
    plt.close()
    
    # scatter plot
    plt.scatter(merged_happiness["GDP per capita"], merged_happiness["Happiness score"], color="green")
    plt.title("GDP vs Happiness score")
    plt.xlabel("GDP per capita")
    plt.ylabel("Happiness Score")
    # save scatter plot
    plt.savefig("outputs/gdp_vs_happiness.png")
    logger.info("Scatter plot saved.")
    plt.close()
    
    # heatmap
    plt.figure(figsize=(10,8))
    numeric_df = merged_happiness.select_dtypes(include="number") 
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    # save heatmap
    plt.savefig("outputs/correlation_heatmap.png")
    logger.info("Heatmap saved.")
    plt.close()
    
# --- TASK 4 ---
@task
def hypothesis_test(merged_happiness):
    logger = get_run_logger()
    
    # test 1
    score_1 = merged_happiness.loc[merged_happiness["year"] == 2019, "Happiness score"]
    score_2 = merged_happiness.loc[merged_happiness["year"] == 2020, "Happiness score"]
    
    t_stat, p_val = stats.ttest_ind(score_1, score_2)
    
    logger.info(f"t-statistic: {t_stat:.3f}")
    logger.info(f"p-value: {p_val:.6f}")
    logger.info(f"Happiness score mean 2019: {score_1.mean():.3f}")
    logger.info(f"Happiness score mean 2020: {score_2.mean():.3f}")
    
    alpha = 0.05
    
    if p_val < alpha:
        logger.info("The analysis suggests that average happiness scores differed between 2019 and 2020.")
    else:
        logger.info("The analysis did not find enough evidence that average happiness scores changed between 2019 and 2020.")
        
    # test 2 - East Asia vs Southeast Asia
    region_1 = merged_happiness.loc[merged_happiness["Regional indicator"] == "East Asia", "Happiness score"]
    region_2 = merged_happiness.loc[merged_happiness["Regional indicator"] == "Southeast Asia", "Happiness score"]
    
    t_stat2, p_val2 = stats.ttest_ind(region_1, region_2)
    
    logger.info(f"East Asia vs Southeast Asia t-statistic: {t_stat2:.3f}")
    logger.info(f"East Asia vs Southeast Asia p-value: {p_val2:.6f}")
    logger.info(f"Mean happiness East Asia: {region_1.mean():.3f}")
    logger.info(f"Mean happiness Southeast Asia: {region_2.mean():.3f}")
    
    alpha = 0.05
    
    if p_val2 < alpha:
        logger.info("The analysis suggests that average happiness scores differed between East Asia and Southeast Asia.")
    else:
        logger.info("The analysis did not find enough evidence that average happiness scores differ between East Asia and Southeast Asia.")
        
    return {
        "t_stat_2019_2020": t_stat,
        "p_val_2019_2020": p_val,
        "t_stat_regions": t_stat2,
        "p_val_regions": p_val2
    }

# --- TASK 5 ---
@task
def corr_and_compare(merged_happiness):
    logger = get_run_logger()
    
    happiness_score = merged_happiness["Happiness score"]
    
    numeric_df = merged_happiness.select_dtypes(include="number") 
    
    target = "Happiness score"
    
    variables = [col for col in numeric_df.columns if col != target]
    number_of_tests = len(variables)
    
    alpha = 0.05
    adjusted_alpha = alpha / number_of_tests
    
    results = []
    
    for variable in variables:
        x = merged_happiness[variable]
        y = happiness_score
        
        r, p = pearsonr(x, y)
        
        logger.info(f"{variable} -> r: {r:.3f}, p: {p:.5f}")
        
        sig_05 = p < alpha
        sig_bonf = p < adjusted_alpha
        
        logger.info(f"  significant (0.05): {sig_05}")
        logger.info(f"  significant (bonferroni): {sig_bonf}")
            
        results.append({
            "variable": variable,
            "r": r,
            "p": p,
            "sig_05": sig_05,
            "sig_bonf": sig_bonf
        })
        
    return results
        
# --- TASK 6 ---
@task
def summary_report(merged_happiness, t_test_results, corr_results):
    logger = get_run_logger()
    
    # total number of countries
    num_of_countries = merged_happiness["Country"].nunique()
    logger.info(f"Number of countries: {num_of_countries}")
    
    # total number of years
    num_of_years = merged_happiness["year"].nunique()
    logger.info(f"Number of years in dataset: {num_of_years}")
    
    # top/bottom 3 regions
    region_means = merged_happiness.groupby("Regional indicator")["Happiness score"].mean()
    
    sorted_regions = region_means.sort_values(ascending=False)
    
    top_3 = sorted_regions.head(3)
    bottom_3 = sorted_regions.tail(3)
    
    logger.info(f"Top 3 happiest regions:\n{top_3}")
    logger.info(f"Bottom 3 happiest regions:\n{bottom_3}")
    
    # t-test interpretation (2019 vs 2020)
    t_stat = t_test_results["t_stat_2019_2020"]
    p_val = t_test_results["p_val_2019_2020"]

    score_2019 = merged_happiness.loc[merged_happiness["year"] == 2019, "Happiness score"]
    score_2020 = merged_happiness.loc[merged_happiness["year"] == 2020, "Happiness score"]

    mean_2019 = score_2019.mean()
    mean_2020 = score_2020.mean()

    logger.info(f"2019 mean: {mean_2019:.3f}")
    logger.info(f"2020 mean: {mean_2020:.3f}")
    logger.info(f"t-statistic: {t_stat:.3f}")
    logger.info(f"p-value: {p_val:.6f}")

    if p_val < 0.05:
        logger.info(
            "There was a statistically significant difference in happiness between 2019 and 2020."
        )
    else:
        logger.info(
            "There was no statistically significant difference in happiness between 2019 and 2020."
        )

    # strongest correlation
    significant = [r for r in corr_results if r["sig_bonf"]]

    if significant:
        strongest = max(significant, key=lambda x: abs(x["r"]))

        logger.info(
            f"Strongest correlation after Bonferroni correction: "
            f"{strongest['variable']} (r = {strongest['r']:.3f})"
        )
    else:
        logger.info("No variables remained significant after Bonferroni correction.")

    return {
        "num_countries": num_of_countries,
        "num_years": num_of_years,
        "t_test_2019_2020": {
            "t_stat": t_stat,
            "p_val": p_val,
            "mean_2019": mean_2019,
            "mean_2020": mean_2020
        },
        "strongest_corr": strongest if significant else None
    }

# pipeline
@flow
def happiness_pipeline():
    all_data = load_data()
    get_stats(all_data)
    create_visuals(all_data)
    t_test_results = hypothesis_test(all_data)
    corr_results = corr_and_compare(all_data)
    summary_report(all_data, t_test_results, corr_results)
    
if __name__ == "__main__":
    happiness_pipeline()
    
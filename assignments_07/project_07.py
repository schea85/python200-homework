from dotenv import load_dotenv
from openai import OpenAI
import os
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent


# --- Pre-Task: Load the Data ---
DATA_PATH = "assignments_01/outputs/merged_happiness.csv"

# --- Task 1: Define Tools ---
df = None

# load_happiness_data tool
@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.
    
    Returns: 
        A dict with "shape" and "columns"
    """
    
    global df
    
    if Path(DATA_PATH).exists():
        df = pd.read_csv(DATA_PATH)
    else:
    # loop
        years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        happiness_dfs = []
        
        for year in years:
            file = f"assignments_01/csv/world_happiness_{year}.csv"
            yearly_df = pd.read_csv(file, sep=";", decimal=",")
            yearly_df["year"] = year
                
            if 'Ladder score' in yearly_df.columns:
                yearly_df = yearly_df.rename(columns={'Ladder score': 'Happiness score'})
                    
            happiness_dfs.append(yearly_df)
                
        # concat dfs
        df = pd.concat(happiness_dfs, ignore_index=True)
    
    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }
    
# summarize_column tool
@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.
    
    Args:
        column: The name of the column to summarize.
        
    Returns:
        A dictionary containing descriptive statistics for the column.
    """
    
    if df is None:
        return {"error": "No data is loaded."}
    
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    
    return df[column].describe().to_dict()

# compute_correlation tool
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.
    
    Args:
        col1: The name of the first numeric column.
        col2: The name of the second numeric column.
        
    Returns:
        A dictionary containing the column names, Pearson correlation coefficient, and p-value.
    """
    
    if df is None:
        return {"error": "No data is loaded."}
            
    if col1 not in df.columns:
        return {"error": f"'{col1}' is not a column."}
            
    if col2 not in df.columns:
        return{"error": f"'{col2}' is not a column."}
    
    
    pearson_r, p_value= pearsonr(df[col1], df[col2])
        
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(pearson_r, 4),
        "p_value": round(p_value, 4)
    }
    
# get_top_n_countries tool
@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.
    
    Args:
        column: The column to use for ranking countries.
        year: The year to filter the data by.
        n: The number of top countries to return.
        
    Returns:
        A dictionary containing the top N countries and their values.
    """
    
    if df is None:
        return {"error": "no data is loaded."}
    
    if column not in df.columns:
        return {"error": f"'{column}' is not a column."}
    
    if "year" not in df.columns:
        return {"error": "Year column not found."}
    
    filtered = df[df["year"] == year]
    
    if filtered.empty:
        return {"error": f"No data found for {year}."}
    
    top_countries = filtered.sort_values(column, ascending=False).head(n)
    
    return {
        "countries": top_countries[["country", column]].to_dict("records")
    }
    

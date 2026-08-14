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



if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")
api_key = os.getenv("OPENAI_API_KEY")

Path("outputs").mkdir(exist_ok=True)

# --- Pre-Task: Load the Data ---
DATA_PATH = "../assignments_01/outputs/merged_happiness.csv"

# --- Task 1: Define Tools ---
df = None

# load_happiness_data tool
@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory and make it the active dataset.
    
    Loads the merged World Happiness CSV from DATA_PATH. If the merged
    file does not exist, loads and merges the yearly CSV files from
    assignments_01/csv/.
    
    Returns: 
        A dict with "shape" and "columns" of the dataframe.
    """
    
    global df
    
    if Path(DATA_PATH).exists():
        df = pd.read_csv(DATA_PATH)
    
    else:
    # loop
        years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        happiness_dfs = []
        
        for year in years:
            file = f"../assignments_01/csv/world_happiness_{year}.csv"
            yearly_df = pd.read_csv(file, sep=";", decimal=",")
            yearly_df["year"] = year
                
            if 'Ladder score' in yearly_df.columns:
                yearly_df = yearly_df.rename(columns={'Ladder score': 'Happiness score'})
                    
            happiness_dfs.append(yearly_df)
                
        # concat dfs
        df = pd.concat(happiness_dfs, ignore_index=True)
    
    return {
        "shape": df.shape,
        "columns": list(df.columns)
    }
    
# summarize_column tool
@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.
    
    Args:
        column (str): The name of the column to summarize.
        
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
    
    clean_df = df[[col1, col2]].dropna()
    
    pearson_r, p_value= pearsonr(clean_df[col1], clean_df[col2])
        
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
        "countries": top_countries[["Country", column]].to_dict("records")
    }
    
# --- Task 2: Build the Agent ---
model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

# --- Task 3: Run Guided Queries ---
queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
]

for query in queries:
    print(f"\n--- Query: {query} ---")
    response = agent.run(query, reset=False)
    print(response)
    
# --- Task 4: My Own Questions ---
my_query_1 = "Using the loaded dataset, what was the average happiness score for 2019?"
response_1 = agent.run(my_query_1, reset=False)
print(response_1)
# Comments:
# The agent generated its own code instead of using the loaded
# dataset and mostly hallucinated.
# The agent created a mock dataset after encountering an error;
# so the result not based on the actual World Happiness data.

my_query_2 = "What are the top 3 happiest countries in 2019?"
response_2 = agent.run(my_query_2, reset=False)
print(response_2)
# Comments: 
# The agent was able to answer this question correctly.
# It used the available tools to get the answer.

# --- Task 5: Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#    significant? Did it use the p-value correctly? What threshold did it apply?
#
#    The agent used the compute_correlation tool to calculate the Pearson correlation
#    coefficient and p-value. It used the p-value to determine if the correlation was statistically 
#    significant and applied the standard/default 0.05 threshold. 
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.
#
#    The agent was mostly able to answer the questions in the query set correctly.
#    It used the available tools to obtain the correct answers.
#    The agent was able to generate a plot but incorrectly. Giving it a plot tool would
#    help. Or maybe add more rules to the system_prompt, like not making up answers. 
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    (You do not need to implement it.)
#
#    An additional tool that would be useful is a tool for finding the average
#    happiness score by region.  The region is listed in the csv, under regional indicator,
#    however, agent has a hard figuring that out. So I should, either manually change the column's name
#    or write an additional tool. This would make it easier for the agent to compare regions and 
#    answer questions about regional trends.

# --- Running the Project ---
if __name__ == "__main__":
    
    # Task 3: Guided Queries
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)
        
    # Task 4: My Queries
    my_query_1 = "Using the loaded dataset, what was the average happiness score for 2019?"
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)

    my_query_2 = "What are the top 3 happiest countries in 2019?"
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
        

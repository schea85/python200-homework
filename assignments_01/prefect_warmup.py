import pandas as pd
import numpy as np
from prefect import task, flow

# Pipeline Q2:

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# step 1
@task
def create_series(arr):
    values = pd.Series(arr, name="values")
    return values

# step 2
@task
def clean_data(series):
    df_cleaned =  series.dropna()
    return df_cleaned

# step 3
@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    
# pipeline
@flow
def pipeline_flow(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary
    
if __name__ == "__main__":
    pipeline_flow(arr)
    
# This pipeline is simple -- just three small functions on a handful of numbers.
# Why might Prefect be more overhead than it is worth here?
#       Because this pipeline is very small, the extra setup/work with Prefect does not
#       add much benefit when using on small, simple pipeline.

# Describe some realistic scenarios where a framework like Prefect could still be useful, 
# even if the pipeline logic itself stays simple like in this case.
#       Prefect is useful for larger workflows, such as processing large datasets, running scheduled ETL jobs, 
#       machine learning pipelines, handling task retries after failures, monitoring workflow execution,
#       or coordinating multiple dependent tasks across different systems.
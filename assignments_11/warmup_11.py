from prefect import task, flow
from prefect.logging import get_run_logger

# --- Prefect Orchestration ---

# Prefect Q1:
# The difference between @task and @flow in Prefect, is that @task is used to define an
# individual unit of work.  Tasks can be tracked, retried, cached, and monitored by Prefect.
# @flow is used to define the overall workflow that coordinates tasks and controls the order
# in which they run.
# No, a pure, in-memory Celsius-to-Fahrenheit calculation does not need to be decorated with @task
# because it is simple, has no I/O, and does not benefit from Prefect's task features such as
# retries or monitoring. It can remain a regular Python helper function.

# Prefect Q2:
@task(retries=3, retry_delay_seconds=30)

# Prefect Q3:
# I would open the failed flow run in the Prefect UI and select the transform task.
# In the task-run details, I would check the logs and error/state information.
# I would expect to see the error message or traceback, when the task failed, any retry
# attempts, and the logs leading up to the failure. Since transform failed, load_enriched
# did not run.

# --- Production Patterns ---

# Production Q1:
# raise_for_status() checks whether the API request was successful. If the server returns
# an error like 500, it raises an exception and causes the task to fail.
# This is better than just printing an error because print() does not stop the pipeline.
# The next tasks could still run with bad or missing data.
# With raise_for_status(), Prefect knows the task failed, can retry it, and will stop dependent
# downstream tasks from running.
# With the print() approach, the pipeline keeps going even though the API failed,
# which could cause problems or bad data in later tasks.

# Production Q2:
# Upsert protects the pipeline from creating duplicate records when we re-run it from the
# beginning. If a record with the same date already exists, it updates that record
# instead of trying to create another one.
# If we used plain insert, the records that were already loaded before the crash
# would still be in the database. Re-running the pipeline would try to insert
# those same records again, which could cause duplicate data or a database error
# because the date already exists.

# Production Q3:
@task
def load_enriched(enrichment_records: list):
    logger = get_run_logger()
    logger.info(f"Upserted {len(enrichment_records)} enrichment records")

# Production Q4:
# The incremental processing check helps make the pipeline idempotent because it only
# processes records that have not already been enriched. This mean running the pipeline
# again does not repeatedly process the same records.
# If we removed the check, the ML and LLM steps would run on all 365 records every time.
# This would increase API/compute costs and make the pipeline take much longer to finish.
# It could also cause data correctness problems because existing enrichment results
# could be unnecessarily overwritten or changed each time the pipeline runs.
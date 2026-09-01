import time

# --- ML vs. LLM in Pipelines ---

# ML/LLM Q1:
# The ML classifier looks at the weather data and produces a binary good/skip prediction
# and a probability. The LLM takes that result and produces a recommendation 
# in human-readable text.
# The ML model is used for the prediction because it is consistent and works well with numeric
# data. The LLM is used for the recommendation because it is good at understanding and generating
# natural language.
# If we used the LLM to make prediction, the results could be less consistent and more expensive.
# If we used the ML model to write the recommendation, it would not be able to produce a useful
# readable sentence.

# ML/LLM Q2:
# a.) converting date string to day-of-week: deterministic code (datetime), because the result is based on 
# fixed rule.

# b.) classifying a job posting: LLM, because it needs to understand freeform text.

# c.) predicting customer churn: trained ML model, because it can learn from labeled numeric data.

# d.) normalizing inconsistent city names: deterministic code, because the names can be mapped
# to standard values.

# e.) revenue total: deterministic code, because it is simple calculation.

# ML/LLM Q3:
# Incremental processing means only processing new records instead of all the records every time
# the pipeline runs.
# It is important because it saves money and time. Re-processing all 365 records could waste
# resources and cause duplicate or incorrect data.

# --- Prompt Design ---

# Prompt Q1:
SYSTEM_PROMPT = (
    "Give a two-sentence recommendation. The first sentence should state the prediction, and the "
    "second sentence should explain why."
    "Be direct and practical. Do not use bullet points or headers."
)

# The validation logic would need to be changed from expecting one sentence
# to allowing exactly two sentences. Any sentence-count or length validation should be adjusted
# to validate the two-part response.

# Prompt Q2:
def call_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None

# I would use this in a production pipeline when an API call might fail
# temporarily, so the pipeline can retry instead of stopping immediately.
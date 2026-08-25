import os
from dotenv import load_dotenv
from supabase import create_client

# --- Supabase Connection ---

# Connection Q1:
# Project URL and API key (anon key) are the two pieces of information supabase-py
# needs to connect to project.
# You can find the information in the Supabase dashboard under the project's API
# settings.
# It should never be hardcoded into script because they can be scraped by bots within 
# minutes. The standard practice is to store them in an .env file and .gitignore.

# Connection Q2:
def get_client():
    load_dotenv()
    
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    if not SUPABASE_URL:
        raise ValueError("Missing SUPABASE_URL environment variable.")
    
    if not SUPABASE_KEY:
        raise ValueError("Missing SUPABASE_KEY environment variable.")
    
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        return supabase
    except:
        raise ValueError("Unable to connect to Supabase.")
    
supabase = get_client()
         
# Connection Q3:
# Row Level Security (RLS) is a security feature that controls who can
# access or change specific rows in a database table.
# RLS is disabled for this course because only working with a single trusted user
# and do not have an authentication layer.
# In a real-world multi-user application, RLS should be enabled when users
# need to access only their own data, such as a SaaS application
# with separate records for each user.

# --- supabase-py CRUD ---

# CRUD Q1:
def insert_test_record(supabase):
    record = {
        "date": "2026-08-24",
        "temperature_2m_max": 12.3,
        "temperature_2m_min": 6.1,
        "precipitation_sum": 0.7,
        "wind_speed_10m_max": 16.9
    }
    
    response = supabase.table("weather_raw").insert(record).execute()
    print(response.data)

# insert_test_record(supabase)

# If I run this function twice, it will insert two records with the same data.
# Since date is the primary key, the second insert would violate the unique constraint.
# To make it safe to run multiple times, I could use upsert() instead of insert().
    
# CRUD Q2:
def get_records_by_date_range(supabase, start, end):
    response_range = supabase.table("weather_raw").select("*").gte("date", start).lte("date", end).execute()
    return response_range.data

result = get_records_by_date_range(supabase, "2026-08-24", "2026-08-24")
print(result)

# CRUD Q3:
# insert() adds new records to the table. If a record with the same primary key
# already exists, the insert will fail. For example, I would use insert when adding a
# completely new weather record that does not already exist in the table.
# upsert() can add a new record or update an existing record if the conflict key
# already exists. For example, I would use upsert when updating weather data for a date 
# that may already exist in the table.

records = [
    {"date": "2026-08-22", "temperature_2m_max": 14.1, "temperature_2m_min": 8.7, "precipitation_sum": 1.2, "wind_speed_10m_max": 20.6},
    {"date": "2026-08-23", "temperature_2m_max": 12.3, "temperature_2m_min": 9.9, "precipitation_sum": 0.0, "wind_speed_10m_max": 12.8},
    {"date": "2026-08-24", "temperature_2m_max": 13.4, "temperature_2m_min": 9.0, "precipitation_sum": 0.5, "wind_speed_10m_max": 15.0},
]

def safe_upsert(supabase, records):
    upsert_response = supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
    print(f"Upserted {len(upsert_response.data)} rows")

upsert = safe_upsert(supabase, records)

# --- Idempotency ---

# Idempotency Q1:
# Idempotency means that running an operation multiple times has the same effect as running
# it once. It matters because data pipelines can fail, retry, and need to be run again.
# For example, if a script uses insert() and crashes halfway through, then is restarted,
# it may try to insert rows that were already added. This could cause errors on existing rows
# or create duplicate records.
# Using upsert or checking for existing records first can make the pipeline safe to run again.
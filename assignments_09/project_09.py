import requests
import os
from dotenv import load_dotenv
from supabase import create_client



# --- Step 1: Extract ----

# fetch
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 38.90,
    "longitude": -77.04,
    "start_date": "2020-01-01",
    "end_date": "2020-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max"
    ],
    "timezone": "America/New_York",
}

response = requests.get(url, params=params)
response.raise_for_status()
data = response.json()

# print summary of data received from API
print(f"Received {len(data['daily']['time'])} days of weather data.")
print(f"Data range: {data['daily']['time'][0]} to {data['daily']['time'][-1]}")
print(f"Columns: {list(data['daily'].keys())}")

# --- Step 2: Transform ---

# transform response into records
daily = data["daily"]

records = [
    {
        "date": daily["time"][i],
        "temperature_2m_max": daily["temperature_2m_max"][i],
        "temperature_2m_min": daily["temperature_2m_min"][i],
        "precipitation_sum": daily["precipitation_sum"][i],
        "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
    }
    for i in range(len(daily["time"]))
]

# handle missing values
records = [r for r in records if all(v is not None for v in r.values())]
print(f"Records after dropping nulls: {len(records)}")

# print 1st and last record
print(f"First record: {records[0]}")
print(f"Last record: {records[-1]}")

# A full year normally has 365 records.  However, since 2020 is a leap year,
# I expect 366 records which is what I received.
# A difference could be caused by a leap year, an incomplete date range,
# or missing data from API.

# --- Step 3: Load ---

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

response = (
    supabase.table("weather_raw")
    .upsert(records, on_conflict="date")
    .execute()
)

print(f"Upserted {len(response.data)} rows in weather_raw")

# Running the script twice does not increase the number of rows because
# upsert updates existing records instead of duplicates.
# This shows that the load step is idempotent and is safe to run
# multiple times.

# --- Step 4: Verify ---

# row count
count_response = supabase.table("weather_raw").select("date", count="exact").execute()
print(f"Rows in weather_raw: {count_response.count}")

# first and last record
first = supabase.table("weather_raw").select("*").eq("date", "2020-01-01").execute()
last = supabase.table("weather_raw").select("*").eq("date", "2020-12-31").execute()
july_fourth = supabase.table("weather_raw").select("*").eq("date", "2020-07-04").execute()

print("First record:", first.data)
print("Last record:", last.data)
print("July 4th:", july_fourth.data)
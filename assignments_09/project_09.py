import requests



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

# transforming response into records
daily = data["daily"]

records = [
    {
        "date": daily["time"][i],
        "temperature_2m_max": daily["temperature_2m_max"][i],
        "temperature_2m_min": daily["temperature_2m_min"][i],
        "precipitation_sum": daily["precipitation_sum"][i],
        "wind_speed_10m_max": daily["precipitation_sum"][i],
    }
    for i in range(len(daily["time"]))
]

print(f"Prepared {len(records)} records")
print("First record:", records[0])
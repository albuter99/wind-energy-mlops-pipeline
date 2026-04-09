import json
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


RAW_DIR = Path("artifacts/raw")
PROCESSED_DIR = Path("artifacts/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_weather_data():
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 57.048,
        "longitude": 9.9187,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
            "cloud_cover",
            "wind_speed_80m",
            "wind_direction_80m",
            "wind_gusts_10m"
        ],
        "timezone": "Europe/Copenhagen",
        "past_days": 92,
        "forecast_days": 7
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }

    hourly_data["temperature_2m"] = hourly.Variables(0).ValuesAsNumpy()
    hourly_data["relative_humidity_2m"] = hourly.Variables(1).ValuesAsNumpy()
    hourly_data["precipitation"] = hourly.Variables(2).ValuesAsNumpy()
    hourly_data["surface_pressure"] = hourly.Variables(3).ValuesAsNumpy()
    hourly_data["cloud_cover"] = hourly.Variables(4).ValuesAsNumpy()
    hourly_data["wind_speed_80m"] = hourly.Variables(5).ValuesAsNumpy()
    hourly_data["wind_direction_80m"] = hourly.Variables(6).ValuesAsNumpy()
    hourly_data["wind_gusts_10m"] = hourly.Variables(7).ValuesAsNumpy()

    df = pd.DataFrame(hourly_data)
    df["location"] = "Aalborg"

    return df


def save_raw_outputs(df: pd.DataFrame):
    raw_json_path = RAW_DIR / "weather_raw.json"
    raw_csv_path = PROCESSED_DIR / "weather_initial.csv"

    df.to_json(raw_json_path, orient="records", indent=2, date_format="iso")
    df.to_csv(raw_csv_path, index=False)

    print(f"Saved raw JSON to: {raw_json_path}")
    print(f"Saved initial CSV to: {raw_csv_path}")


if __name__ == "__main__":
    df = fetch_weather_data()
    print(df.head())
    print(df.shape)
    save_raw_outputs(df)

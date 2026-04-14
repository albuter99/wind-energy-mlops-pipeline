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

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_80m",
    "wind_direction_80m",
    "wind_gusts_10m"
]


def get_client():
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def build_dataframe(response):
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


def fetch_historical_data():
    client = get_client()

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 57.048,
        "longitude": 9.9187,
        "hourly": HOURLY_VARIABLES,
        "timezone": "Europe/Berlin",
        "past_days": 92,
        "forecast_days": 0
    }

    response = client.weather_api(url, params=params)[0]
    return build_dataframe(response)


def fetch_forecast_data():
    client = get_client()

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 57.048,
        "longitude": 9.9187,
        "hourly": HOURLY_VARIABLES,
        "timezone": "Europe/Copenhagen",
        "forecast_days": 7
    }

    response = client.weather_api(url, params=params)[0]
    return build_dataframe(response)


def save_outputs(df: pd.DataFrame, raw_name: str, csv_name: str):
    raw_path = RAW_DIR / raw_name
    csv_path = PROCESSED_DIR / csv_name

    df.to_json(raw_path, orient="records", indent=2, date_format="iso")
    df.to_csv(csv_path, index=False)

    print(f"Saved raw JSON to: {raw_path}")
    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    historical_df = fetch_historical_data()
    forecast_df = fetch_forecast_data()

    save_outputs(
        historical_df,
        "weather_historical_raw.json",
        "weather_historical.csv"
    )

    save_outputs(
        forecast_df,
        "weather_forecast_raw.json",
        "weather_forecast.csv"
    )

    print("Historical shape:", historical_df.shape)
    print("Forecast shape:", forecast_df.shape)
    print("\nHistorical head:")
    print(historical_df.head())
    print("\nForecast head:")
    print(forecast_df.head())

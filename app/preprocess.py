from pathlib import Path

import pandas as pd


# INPUTS
HIST_INPUT_PATH = Path("artifacts/processed/weather_historical.csv")
FORECAST_INPUT_PATH = Path("artifacts/processed/weather_forecast.csv")

# OUTPUTS
HIST_OUTPUT_PATH = Path("artifacts/processed/weather_historical_clean.csv")
FORECAST_OUTPUT_PATH = Path("artifacts/processed/weather_forecast_clean.csv")


def preprocess_weather_data(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()

    # Convert date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Numeric columns
    numeric_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "surface_pressure",
        "cloud_cover",
        "wind_speed_80m",
        "wind_direction_80m",
        "wind_gusts_10m",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort
    df = df.sort_values("date").reset_index(drop=True)

    print(f"\n[{name}] Missing values before cleaning:")
    print(df.isna().sum())

    # Fill missing values forward, then backward
    df = df.ffill().bfill()

    print(f"\n[{name}] Missing values after cleaning:")
    print(df.isna().sum())

    return df


if __name__ == "__main__":
    # Load datasets
    hist_df = pd.read_csv(HIST_INPUT_PATH)
    forecast_df = pd.read_csv(FORECAST_INPUT_PATH)

    # Preprocess
    hist_clean = preprocess_weather_data(hist_df, "HISTORICAL")
    forecast_clean = preprocess_weather_data(forecast_df, "FORECAST")

    # Save outputs
    HIST_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    hist_clean.to_csv(HIST_OUTPUT_PATH, index=False)
    forecast_clean.to_csv(FORECAST_OUTPUT_PATH, index=False)

    print(f"\nSaved historical clean data to: {HIST_OUTPUT_PATH}")
    print(f"Saved forecast clean data to: {FORECAST_OUTPUT_PATH}")

    print("\nHistorical shape:", hist_clean.shape)
    print("Forecast shape:", forecast_clean.shape)

    print("\nHistorical head:")
    print(hist_clean.head())

    print("\nForecast head:")
    print(forecast_clean.head())

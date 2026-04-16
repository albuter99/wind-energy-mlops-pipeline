from pathlib import Path

import pandas as pd


# INPUTS
HIST_INPUT_PATH = Path("artifacts/processed/weather_historical_clean.csv")
FORECAST_INPUT_PATH = Path("artifacts/processed/weather_forecast_clean.csv")

# OUTPUTS
HIST_OUTPUT_PATH = Path("artifacts/features/weather_historical_features.csv")
FORECAST_OUTPUT_PATH = Path("artifacts/features/weather_forecast_features.csv")
HIST_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def compute_theoretical_energy(wind_speed: float) -> float:
    if pd.isna(wind_speed):
        return pd.NA
    if wind_speed < 3:
        return 0.0
    elif wind_speed < 12:
        return ((wind_speed - 3) / 9) ** 3
    elif wind_speed < 25:
        return 1.0
    else:
        return 0.0


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Time features
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # Lag features
    df["wind_speed_80m_lag_1"] = df["wind_speed_80m"].shift(1)
    df["wind_speed_80m_lag_2"] = df["wind_speed_80m"].shift(2)
    df["wind_speed_80m_lag_3"] = df["wind_speed_80m"].shift(3)
    df["wind_speed_80m_lag_24"] = df["wind_speed_80m"].shift(24)

    # Rolling features
    df["wind_speed_80m_roll_3"] = df["wind_speed_80m"].rolling(3).mean()
    df["wind_speed_80m_roll_6"] = df["wind_speed_80m"].rolling(6).mean()

    # Interaction / nonlinear features
    df["wind_speed_80m_cubed"] = df["wind_speed_80m"] ** 3
    df["temp_wind_interaction"] = df["temperature_2m"] * df["wind_speed_80m"]

    # Theoretical target
    df["theoretical_energy"] = df["wind_speed_80m"].apply(compute_theoretical_energy)
    df["target_energy_next_hour"] = df["theoretical_energy"].shift(-1)

    return df


if __name__ == "__main__":
    # Load cleaned historical and forecast data
    hist_df = pd.read_csv(HIST_INPUT_PATH)
    forecast_df = pd.read_csv(FORECAST_INPUT_PATH)

    # Keep original lengths to split later
    hist_len = len(hist_df)
    forecast_len = len(forecast_df)

    # Combine to generate correct lags/rolling for forecast
    combined_df = pd.concat([hist_df, forecast_df], ignore_index=True)
    combined_features = create_features(combined_df)

    # Split again
    hist_features = combined_features.iloc[:hist_len].copy()
    forecast_features = combined_features.iloc[hist_len:hist_len + forecast_len].copy()

    # Historical data needs a valid next-hour target
    hist_features = hist_features.dropna().reset_index(drop=True)

    # Forecast data should keep the first future rows for inference
    # Remove only rows with missing feature columns, but keep target if it exists or not
    forecast_feature_columns = [
        "date",
        "location",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "surface_pressure",
        "cloud_cover",
        "wind_speed_80m",
        "wind_direction_80m",
        "wind_gusts_10m",
        "hour",
        "day_of_week",
        "month",
        "wind_speed_80m_lag_1",
        "wind_speed_80m_lag_2",
        "wind_speed_80m_lag_3",
        "wind_speed_80m_lag_24",
        "wind_speed_80m_roll_3",
        "wind_speed_80m_roll_6",
        "wind_speed_80m_cubed",
        "temp_wind_interaction",
        "theoretical_energy",
        "target_energy_next_hour",
    ]

    forecast_features = forecast_features.dropna(
        subset=[
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
            "cloud_cover",
            "wind_speed_80m",
            "wind_direction_80m",
            "wind_gusts_10m",
            "hour",
            "day_of_week",
            "month",
            "wind_speed_80m_lag_1",
            "wind_speed_80m_lag_2",
            "wind_speed_80m_lag_3",
            "wind_speed_80m_lag_24",
            "wind_speed_80m_roll_3",
            "wind_speed_80m_roll_6",
            "wind_speed_80m_cubed",
            "temp_wind_interaction",
            "theoretical_energy",
        ]
    ).reset_index(drop=True)

    # Save outputs
    hist_features.to_csv(HIST_OUTPUT_PATH, index=False)
    forecast_features.to_csv(FORECAST_OUTPUT_PATH, index=False)

    print(f"Saved historical features to: {HIST_OUTPUT_PATH}")
    print(f"Saved forecast features to: {FORECAST_OUTPUT_PATH}")

    print("\nHistorical features shape:", hist_features.shape)
    print("Forecast features shape:", forecast_features.shape)

    print("\nHistorical features head:")
    print(hist_features.head())

    print("\nForecast features head:")
    print(forecast_features.head())

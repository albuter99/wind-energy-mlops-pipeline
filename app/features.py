from pathlib import Path

import pandas as pd


INPUT_PATH = Path("artifacts/processed/weather_clean.csv")
OUTPUT_PATH = Path("artifacts/features/weather_features.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


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

    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    df["wind_speed_80m_lag_1"] = df["wind_speed_80m"].shift(1)
    df["wind_speed_80m_lag_2"] = df["wind_speed_80m"].shift(2)
    df["wind_speed_80m_lag_3"] = df["wind_speed_80m"].shift(3)
    df["wind_speed_80m_lag_24"] = df["wind_speed_80m"].shift(24)

    df["wind_speed_80m_roll_3"] = df["wind_speed_80m"].rolling(3).mean()
    df["wind_speed_80m_roll_6"] = df["wind_speed_80m"].rolling(6).mean()

    df["wind_speed_80m_cubed"] = df["wind_speed_80m"] ** 3
    df["temp_wind_interaction"] = df["temperature_2m"] * df["wind_speed_80m"]

    df["theoretical_energy"] = df["wind_speed_80m"].apply(compute_theoretical_energy)
    df["target_energy_next_hour"] = df["theoretical_energy"].shift(-1)

    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    features_df = create_features(df)
    features_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved features data to: {OUTPUT_PATH}")
    print(features_df.head())
    print(features_df.shape)

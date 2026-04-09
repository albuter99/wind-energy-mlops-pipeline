from pathlib import Path

import pandas as pd


INPUT_PATH = Path("artifacts/processed/weather_initial.csv")
OUTPUT_PATH = Path("artifacts/processed/weather_clean.csv")


def preprocess_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

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

    df = df.sort_values("date").reset_index(drop=True)

    print("Missing values before cleaning:")
    print(df.isna().sum())

    df = df.dropna().reset_index(drop=True)

    print("Missing values after cleaning:")
    print(df.isna().sum())

    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    clean_df = preprocess_weather_data(df)
    clean_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved cleaned data to: {OUTPUT_PATH}")
    print(clean_df.head())
    print(clean_df.shape)

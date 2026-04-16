import json
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

FEATURE_IMPORTANCE_PATH = Path("artifacts/metrics/feature_importance.json")

FEATURES_PATH = Path("artifacts/features/weather_forecast_features.csv")
PREDICTIONS_PATH = Path("artifacts/predictions/predictions.csv")
OUTPUT_PATH = Path("docs/dashboard_data.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TIMEZONE = ZoneInfo("Europe/Copenhagen")


def build_weather_summary(row: pd.Series) -> str:
    wind_speed = float(row["wind_speed_80m"])
    temperature = float(row["temperature_2m"])
    pressure = float(row["surface_pressure"])
    humidity = float(row["relative_humidity_2m"])
    precipitation = float(row["precipitation"])

    return (
        f"Forecast conditions in Aalborg indicate a wind speed of {wind_speed:.2f} m/s, "
        f"temperature of {temperature:.2f} °C, surface pressure of {pressure:.2f} hPa, "
        f"relative humidity of {humidity:.2f}%, and precipitation of {precipitation:.2f} mm. "
        f"These forecast conditions are used to estimate next-hour theoretical wind energy output."
    )


def build_recommendation(row: pd.Series, predicted_energy: float) -> tuple[str, str]:
    wind_speed = float(row["wind_speed_80m"])

    if wind_speed < 3:
        return (
            "Do not activate the turbines. Wind speed is too low for profitable operation.",
            "danger",
        )
    elif wind_speed >= 25:
        return (
            "Do not activate the turbines. Wind speed is too high and may exceed the safe operating range.",
            "danger",
        )
    elif predicted_energy < 0.15:
        return (
            "Activation is not recommended. Predicted theoretical energy is too low.",
            "warning",
        )
    else:
        return (
            "Activate the turbines. Conditions are within the operating range and predicted theoretical energy is sufficient.",
            "success",
        )


def get_next_full_hour(now_dt: datetime) -> datetime:
    rounded = now_dt.replace(minute=0, second=0, microsecond=0)
    return rounded + timedelta(hours=1)


def generate_dashboard_data():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing file: {FEATURES_PATH}")

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {PREDICTIONS_PATH}")

    features_df = pd.read_csv(FEATURES_PATH)
    predictions_df = pd.read_csv(PREDICTIONS_PATH)

    if features_df.empty:
        raise ValueError("forecast features dataset is empty")

    if predictions_df.empty:
        raise ValueError("predictions dataset is empty")

    features_df["date"] = pd.to_datetime(features_df["date"], utc=True)
    predictions_df["date"] = pd.to_datetime(predictions_df["date"], utc=True)

    now_cph = datetime.now(TIMEZONE)
    next_hour_cph = get_next_full_hour(now_cph)
    next_hour_utc = next_hour_cph.astimezone(ZoneInfo("UTC"))

    valid_predictions = predictions_df[predictions_df["date"] >= next_hour_utc].copy()
    valid_features = features_df[features_df["date"] >= next_hour_utc].copy()

    if valid_predictions.empty or valid_features.empty:
        raise ValueError("No valid forecast row found for the next full hour.")

    selected_prediction_row = valid_predictions.iloc[0]
    selected_feature_row = valid_features.iloc[0]

    latest_prediction = float(selected_prediction_row["predicted_energy_next_hour"])
    latest_target = float(selected_prediction_row["target_energy_next_hour"])
    latest_date = str(selected_prediction_row["date"])

    weather_summary = build_weather_summary(selected_feature_row)
    recommendation, recommendation_level = build_recommendation(
        selected_feature_row,
        latest_prediction
    )

    historical = []
    for _, row in predictions_df.iterrows():
        if pd.notna(row["predicted_energy_next_hour"]) and pd.notna(row["target_energy_next_hour"]):
            historical.append({
                "date": str(row["date"]),
                "predicted_energy_next_hour": float(row["predicted_energy_next_hour"]),
                "target_energy_next_hour": float(row["target_energy_next_hour"])
            })

    forecast = []
    for _, row in valid_predictions.head(24).iterrows():
        if pd.notna(row["predicted_energy_next_hour"]) and pd.notna(row["target_energy_next_hour"]):
            forecast.append({
                "date": str(row["date"]),
                "predicted_energy_next_hour": float(row["predicted_energy_next_hour"]),
                "target_energy_next_hour": float(row["target_energy_next_hour"])
            })

    # Load feature importance if available
    feature_importance = []
    if FEATURE_IMPORTANCE_PATH.exists():
        with open(FEATURE_IMPORTANCE_PATH, encoding="utf-8") as f:
            feature_importance = json.load(f)

    dashboard_data = {
        "latest_prediction": latest_prediction,
        "latest_target": latest_target,
        "latest_date": latest_date,
        "weather_summary": weather_summary,
        "recommendation": recommendation,
        "recommendation_level": recommendation_level,
        "last_updated": now_cph.isoformat(),
        "next_full_hour": next_hour_cph.isoformat(),
        "historical": historical,
        "forecast": forecast,
        "wind_speed": float(selected_feature_row["wind_speed_80m"]),  # add this
        "feature_importance": feature_importance}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)

    print(f"Dashboard data saved to: {OUTPUT_PATH}")
    print(json.dumps(dashboard_data, indent=2, ensure_ascii=False)[:1200])


if __name__ == "__main__":
    generate_dashboard_data()

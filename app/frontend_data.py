import json
from pathlib import Path

import pandas as pd


FEATURES_PATH = Path("artifacts/features/weather_features.csv")
PREDICTIONS_PATH = Path("artifacts/predictions/predictions.csv")
OUTPUT_PATH = Path("docs/dashboard_data.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_weather_summary(row: pd.Series) -> str:
    wind_speed = row["wind_speed_80m"]
    temperature = row["temperature_2m"]
    pressure = row["surface_pressure"]
    humidity = row["relative_humidity_2m"]
    precipitation = row["precipitation"]

    return (
        f"Current conditions in Aalborg show a wind speed of {wind_speed:.2f} m/s, "
        f"temperature of {temperature:.2f} °C, surface pressure of {pressure:.2f} hPa, "
        f"relative humidity of {humidity:.2f}%, and precipitation of {precipitation:.2f} mm. "
        f"These conditions are used to estimate next-hour theoretical wind energy output."
    )


def build_recommendation(row: pd.Series, predicted_energy: float) -> str:
    wind_speed = row["wind_speed_80m"]

    if wind_speed < 3:
        return "Do not activate the turbines. Wind speed is too low for profitable operation."
    elif wind_speed >= 25:
        return "Do not activate the turbines. Wind speed is too high and may exceed the safe operating range."
    elif predicted_energy < 0.15:
        return "Activation is not recommended. Predicted theoretical energy is too low."
    else:
        return "Activate the turbines. Conditions are within the operating range and predicted theoretical energy is sufficient."


def generate_dashboard_data():
    features_df = pd.read_csv(FEATURES_PATH)
    predictions_df = pd.read_csv(PREDICTIONS_PATH)

    latest_feature_row = features_df.iloc[-1]
    latest_prediction_row = predictions_df.iloc[-1]

    latest_prediction = float(latest_prediction_row["predicted_energy_next_hour"])
    latest_target = float(latest_prediction_row["target_energy_next_hour"])
    latest_date = str(latest_prediction_row["date"])

    weather_summary = build_weather_summary(latest_feature_row)
    recommendation = build_recommendation(latest_feature_row, latest_prediction)

    historical = []
    for _, row in predictions_df.iterrows():
        historical.append({
            "date": str(row["date"]),
            "predicted_energy_next_hour": float(row["predicted_energy_next_hour"]),
            "target_energy_next_hour": float(row["target_energy_next_hour"])
        })

    forecast = historical[-24:]

    dashboard_data = {
        "latest_prediction": latest_prediction,
        "latest_target": latest_target,
        "latest_date": latest_date,
        "weather_summary": weather_summary,
        "recommendation": recommendation,
        "historical": historical,
        "forecast": forecast
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)

    print(f"Dashboard data saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_dashboard_data()

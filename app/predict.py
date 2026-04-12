import json
from pathlib import Path

import joblib
import pandas as pd


INPUT_PATH = Path("artifacts/features/weather_forecast_features.csv")
MODEL_PATH = Path("artifacts/models/model.pkl")

PREDICTIONS_DIR = Path("artifacts/predictions")
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = PREDICTIONS_DIR / "predictions.csv"
OUTPUT_JSON = Path("docs/predictions.json")
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def run_prediction():
    df = pd.read_csv(INPUT_PATH)
    model = joblib.load(MODEL_PATH)

    if df.empty:
        raise ValueError("Forecast feature dataset is empty. Cannot generate predictions.")

    # Features only
    X = df.drop(columns=[
        "date",
        "location",
        "theoretical_energy",
        "target_energy_next_hour"
    ])

    # Predict all future rows
    predictions = model.predict(X)

    results = df[["date", "location", "target_energy_next_hour"]].copy()
    results["predicted_energy_next_hour"] = predictions

    # Save full forecast predictions
    results.to_csv(OUTPUT_CSV, index=False)
    results.to_json(OUTPUT_JSON, orient="records", indent=2)

    # Next-hour prediction = first future row
    next_hour_result = results.iloc[0].to_dict()

    print(f"Predictions saved to: {OUTPUT_CSV}")
    print(f"Frontend JSON saved to: {OUTPUT_JSON}")

    print("\nNext-hour prediction:")
    print(json.dumps(next_hour_result, indent=2, ensure_ascii=False))

    print("\nForecast preview:")
    print(results.head())


if __name__ == "__main__":
    run_prediction()

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


INPUT_PATH = Path("artifacts/features/weather_historical_features.csv")
MODEL_PATH = Path("artifacts/models/model.pkl")
METRICS_PATH = Path("artifacts/metrics/metrics.json")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)


def train_and_evaluate():
    df = pd.read_csv(INPUT_PATH)

    # Target
    y = df["target_energy_next_hour"]

    # Features
    X = df.drop(columns=[
        "date",
        "location",
        "theoretical_energy",
        "target_energy_next_hour"
    ])

    # Train/test split using only historical data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    results = {}
    best_model_name = None
    best_model = None
    best_rmse = float("inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5

        results[name] = {
            "MAE": round(float(mae), 6),
            "RMSE": round(float(rmse), 6)
        }

        print(f"{name} -> MAE: {mae:.6f}, RMSE: {rmse:.6f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = model

    # Save best model
    joblib.dump(best_model, MODEL_PATH)

    # Save metrics
    output = {
        "training_dataset": str(INPUT_PATH),
        "best_model": best_model_name,
        "results": results
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nBest model: {best_model_name}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    train_and_evaluate()

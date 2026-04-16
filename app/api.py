from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI


app = FastAPI(title="Wind Energy Forecast API")

MODEL_PATH = Path("artifacts/model.pkl")
DATA_PATH = Path("artifacts/features/weather_features.csv")


@app.get("/")
def root():
    return {"message": "Wind Energy Forecast API is running"}


@app.get("/predict")
def predict():
    # Load model
    model = joblib.load(MODEL_PATH)

    # Load latest engineered features
    df = pd.read_csv(DATA_PATH)

    # Keep the latest row for prediction
    latest_row = df.iloc[[-1]].copy()

    prediction_date = latest_row["date"].values[0]

    # Drop non-feature columns
    X_latest = latest_row.drop(columns=[
        "date",
        "location",
        "theoretical_energy",
        "target_energy_next_hour"
    ])

    prediction = model.predict(X_latest)[0]

    return {
        "prediction": round(float(prediction), 6),
        "date": str(prediction_date)
    }

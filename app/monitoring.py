import json
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("artifacts/features/weather_features.csv")
OUTPUT_PATH = Path("artifacts/metrics/monitoring.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def run_monitoring():
    errors = []

    try:
        df = pd.read_csv(INPUT_PATH)
    except Exception as e:
        errors.append(f"Could not load feature dataset: {str(e)}")

        monitoring_output = {
            "status": "failed",
            "errors": errors
        }

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(monitoring_output, f, indent=2, ensure_ascii=False)

        print(f"Monitoring failed. Log saved to: {OUTPUT_PATH}")
        return

    record_count = len(df)
    column_count = len(df.columns)
    missing_values = df.isna().sum().to_dict()

    theoretical_energy_distribution = {
        "min": float(df["theoretical_energy"].min()),
        "max": float(df["theoretical_energy"].max()),
        "mean": float(df["theoretical_energy"].mean()),
        "std": float(df["theoretical_energy"].std()),
    }

    prediction_target_distribution = {
        "min": float(df["target_energy_next_hour"].min()),
        "max": float(df["target_energy_next_hour"].max()),
        "mean": float(df["target_energy_next_hour"].mean()),
        "std": float(df["target_energy_next_hour"].std()),
    }

    monitoring_output = {
        "status": "success",
        "record_count": record_count,
        "column_count": column_count,
        "missing_values": missing_values,
        "theoretical_energy_distribution": theoretical_energy_distribution,
        "target_energy_next_hour_distribution": prediction_target_distribution,
        "errors": errors
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(monitoring_output, f, indent=2, ensure_ascii=False)

    print(f"Monitoring log saved to: {OUTPUT_PATH}")
    print(json.dumps(monitoring_output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_monitoring()

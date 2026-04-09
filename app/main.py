import subprocess
import sys
from pathlib import Path


def run_step(script_name: str):
    script_path = Path("app") / script_name
    print(f"\nRunning {script_path} ...")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        check=True
    )

    if result.returncode == 0:
        print(f"Finished {script_name} successfully.")


def run_pipeline():
    steps = [
    "fetch.py",
    "preprocess.py",
    "features.py",
    "train.py",
    "predict.py",
    "monitoring.py",
]

    for step in steps:
        run_step(step)

    print("\nFull pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()

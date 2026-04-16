import sqlite3
from pathlib import Path

import pandas as pd


DB_PATH = Path("data/weather.db")
INPUT_PATH = Path("artifacts/processed/weather_clean.csv")


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS weather (
        date TEXT,
        location TEXT,
        wind REAL,
        temp REAL,
        precipitation REAL,
        humidity REAL,
        pressure REAL,
        cloud_cover REAL,
        wind_direction REAL,
        wind_gusts REAL
    )
    """)

    conn.commit()
    return conn


def insert_data(conn, df: pd.DataFrame):
    rows = []

    for _, row in df.iterrows():
        rows.append((
            str(row["date"]),
            row["location"],
            float(row["wind_speed_80m"]),
            float(row["temperature_2m"]),
            float(row["precipitation"]),
            float(row["relative_humidity_2m"]),
            float(row["surface_pressure"]),
            float(row["cloud_cover"]),
            float(row["wind_direction_80m"]),
            float(row["wind_gusts_10m"]),
        ))

    conn.executemany("""
    INSERT INTO weather (
        date, location, wind, temp, precipitation,
        humidity, pressure, cloud_cover, wind_direction, wind_gusts
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    conn.commit()
    print(f"Inserted {len(rows)} rows into SQLite database.")


def test_db(conn):
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM weather")
    total_rows = cursor.fetchone()[0]
    print(f"Total rows in weather table: {total_rows}")

    cursor.execute("SELECT * FROM weather LIMIT 5")
    sample_rows = cursor.fetchall()

    print("Sample rows:")
    for row in sample_rows:
        print(row)


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)

    conn = init_db()
    insert_data(conn, df)
    test_db(conn)
    conn.close()

    print(f"Database saved to: {DB_PATH}")

# 🌬️ Wind Energy Prediction MLOps Pipeline

---

## 🚀 Project Overview

This project implements a **production-style MLOps pipeline** to predict next-hour theoretical wind energy output using weather data for Aalborg, Denmark.

The system is designed as a **fully automated, reproducible, and deployable pipeline**, covering the entire ML lifecycle from data ingestion to end-user visualization.

---

## 🎯 Objectives

- Predict **next-hour wind energy output** (normalized 0–1 scale)  
- Provide **operational recommendations** (activate / marginal / do not activate)  
- Demonstrate **end-to-end MLOps principles**  
- Deliver a **live data product (dashboard + API)**  

---

## 🧠 Key MLOps Components

This project follows a complete MLOps architecture:

### 📡 Data Ingestion
- Source: Open-Meteo API  
- Hourly weather data (historical + forecast)  
- Cached & retry-enabled requests  

### 🧹 Data Preprocessing
- Timestamp parsing  
- Sorting & type coercion  
- Missing value handling (forward-fill + backward-fill)  

### ⚙️ Feature Engineering
- Time features: hour, day, month  
- Lag features (1h, 2h, 3h, 24h)  
- Rolling averages (3h, 6h)  
- Wind cubic transformation (physics-inspired)  
- Target: **next-hour energy prediction**  

### 🤖 Model Training
- Linear Regression (baseline)  
- Random Forest (non-linear model)  

Evaluation metrics:
- MAE  
- RMSE  

- Automatic best model selection  

### 🔮 Prediction Pipeline
- Generates next-hour forecasts  
- Produces structured prediction artifacts  
- Feeds dashboard data  

### 📊 Monitoring
- Dataset health tracking  
- Missing values per column  
- Distribution statistics (mean, std, min, max)  
- Stored as versioned artifacts  

### 🌐 Deployment
- FastAPI prediction endpoint  
- Dockerized application  
- GitHub Pages dashboard  

### ⚙️ Automation (CI/CD)
- GitHub Actions pipeline  
- Hourly scheduled runs (cron)  
- Automatic artifact updates  
- Versioned outputs in repository  

---

## 🏗️ System Architecture

```text
Open-Meteo API
        ↓
    fetch.py
        ↓
  preprocess.py
        ↓
   features.py
        ↓
    train.py
        ↓
   predict.py
        ↓
frontend_data.py
        ↓
 GitHub Pages Dashboard

+ Monitoring (monitoring.py)
+ Orchestration (main.py)
+ Automation (GitHub Actions)
```

## 📂 Project Structure

```text
.
├── app/
│   ├── fetch.py              # Data ingestion
│   ├── preprocess.py         # Cleaning
│   ├── features.py           # Feature engineering
│   ├── train.py              # Model training
│   ├── predict.py            # Predictions
│   ├── monitoring.py         # Data monitoring
│   ├── frontend_data.py      # Dashboard JSON generation
│   ├── main.py               # Pipeline orchestrator
│   └── api.py                # FastAPI service
│
├── artifacts/
│   ├── raw/                  # Raw API data
│   ├── processed/            # Clean datasets
│   ├── features/             # Feature tables
│   ├── models/               # Trained models
│   ├── predictions/          # Forecast outputs
│   └── metrics/              # Evaluation & monitoring
│
├── docs/
│   ├── index.html            # Dashboard UI
│   └── dashboard_data.json   # Data consumed by frontend
│
├── .github/workflows/
│   └── pipeline.yml          # CI/CD automation
│
├── Dockerfile                # Containerization
├── requirements.txt
└── README.md
```

## ▶️ How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python app/main.py
```

## 🐳 Run with Docker

### Build image
```bash
docker build -t wind-mlops .
```

### Run container
```bash
docker run -p 8000:8000 wind-mlops
```

## 🔌 API Usage

### Start FastAPI server:
```bash
uvicorn app.api:app --reload
```

### Endpoint
GET /predict

### Returns
- Next-hour prediction  
- Timestamp  

## 🌍 Live Dashboard

👉 GitHub Pages automatically publishes results from /docs

The dashboard includes:

- Next-hour energy prediction  
- Operational recommendation  
- Forecast visualization  
- Predicted vs theoretical comparison  

## 📦 MLOps Artifacts

Type | Description
-----|-------------
Raw Data | API responses
Processed | Cleaned datasets
Features | ML-ready tables
Models | Serialized model (.pkl)
Metrics | MAE / RMSE
Predictions | Forecast outputs
Monitoring | Data quality stats
Frontend | Dashboard JSON

## ⚠️ Limitations & Improvements

### Current Limitations
- Uses simple hold-out validation (not time-series split)
- Monitoring does not include drift detection
- Artifacts stored in Git (not scalable)
- Dashboard mixes forecast & historical data

### Future Improvements
- Walk-forward validation
- Drift detection & alerting
- External artifact storage (S3 / MLflow)
- Real historical backtesting

## 👤 Authors
- Ioannis Chatzikos
- Kristjana Prifti
- Timo Bertus Rik Philipse
- Victor Carmona García
- Álvaro Buendía

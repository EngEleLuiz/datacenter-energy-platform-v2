# Datacenter Energy Intelligence Platform

[![CI](https://github.com/EngEleLuiz/datacenter-energy-platform-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/EngEleLuiz/datacenter-energy-platform-v2/actions)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://datacenter-energy-platform-v2.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end platform for real-time datacenter telemetry, ML-powered anomaly detection, and Grid-Forming/Grid-Following (GFM/GFL) inverter mode classification.

---

## Live Demo

▶ **[datacenter-energy-platform-v2.streamlit.app](https://datacenter-energy-platform-v2.streamlit.app)**

---

## What This Does

| Module | Description |
|---|---|
| **GFM/GFL Classifier** | Detects inverter control mode (Grid-Following, Grid-Forming, Transitioning) from 5-min SCADA telemetry using temporal rolling features |
| **Anomaly Detection** | Isolation Forest on server telemetry — CPU, memory, network, power |
| **PUE Forecasting** | LSTM-based Power Usage Effectiveness forecasting |
| **Real-time Dashboard** | Streamlit dashboard with SHAP explanations, Grafana KPIs, weather/energy pricing |

---

## Architecture

```
┌─────────────────┐     Kafka      ┌──────────────────┐
│  data-generator │ ─────────────► │   ml-worker      │
│  (Container 1)  │                │   (Container 3)  │
│                 │                │   Airflow DAGs   │
│  Inverter sim   │                │   MLflow tracking│
│  Server sim     │                └────────┬─────────┘
│  UPS sim        │                         │ model artifacts
└─────────────────┘                         ▼
                                   ┌──────────────────┐
                                   │    dashboard     │
         PostgreSQL ◄──────────────│   (Container 2)  │
         MinIO      ◄──────────────│   Streamlit      │
         MLflow     ◄──────────────│   Port 8501      │
                                   └──────────────────┘
```

---

## Containers

| Container | Image size | Purpose |
|---|---|---|
| `generator` | ~200 MB | Runs inverter/server/UPS simulators → Kafka |
| `dashboard` | ~600 MB | Streamlit app + ML inference |
| `ml-worker` | ~2 GB | Airflow DAGs + model training (local only) |

---

## Quick Start (Docker)

```bash
# 1. Clone
git clone https://github.com/EngEleLuiz/datacenter-energy-platform-v2
cd datacenter-energy-platform-v2

# 2. Configure secrets
cp .env.example .env
# Edit .env with your passwords (see .env.example for required keys)

# 3. Start infrastructure + generator + dashboard
docker compose up -d postgres minio kafka mlflow grafana generator dashboard

# 4. Open dashboard
open http://localhost:8501

# 5. (Optional) Start Airflow — local training only
docker compose --profile local up -d ml-worker
open http://localhost:8080
```

---

## Streamlit Cloud Deploy

The dashboard runs on [Streamlit Cloud](https://streamlit.io/cloud) with zero config.

**Settings in Streamlit Cloud:**
- Repository: `EngEleLuiz/datacenter-energy-platform-v2`
- Branch: `main`
- Main file path: `dashboard/app.py`
- Python version: `3.11`

Secrets (set in Streamlit Cloud → App Settings → Secrets):
```toml
OPENWEATHER_API_KEY = "your_key_here"
POSTGRES_HOST = "your_db_host"
POSTGRES_USER = "your_user"
POSTGRES_PASSWORD = "your_password"
POSTGRES_DB = "your_db"
```

If secrets are not set, the dashboard runs in **demo mode** (simulates data on-the-fly, no external connections needed).

---

## Project Structure

```
datacenter-energy-platform-v2/
├── data_generator/          # Container 1 — simulators
│   ├── ups_inverter_simulator.py
│   ├── server_simulator.py
│   ├── kafka_producer.py
│   └── requirements.txt     # minimal: kafka, pandas, numpy
├── dashboard/               # Container 2 — Streamlit
│   ├── app.py
│   └── requirements.txt     # streamlit, sklearn, shap, plotly
├── dags/                    # Container 3 — Airflow DAGs
├── docker/                  # One Dockerfile per container
│   ├── data-generator/Dockerfile
│   ├── dashboard/Dockerfile
│   └── ml-worker/Dockerfile
├── ml/                      # Trained model artifacts
│   ├── gfm_classifier.pkl
│   ├── gfm_scaler.pkl
│   └── gfm_features.json
├── notebooks/               # Research (not in any container)
│   ├── 03_gfm_gfl_classifier.ipynb
│   └── 06_nrel_validation.ipynb
├── paper/                   # IEEE submission (LaTeX)
│   └── paper_gfm_gfl_classifier.tex
├── tests/
│   └── test_simulators.py
├── .env.example             # Secret template — copy to .env
├── docker-compose.yml       # All secrets via .env, no hardcoding
└── .gitignore               # Excludes .env, *.pkl, *.csv, fix_*.py
```

---

## Research

The GFM/GFL classifier is documented in a paper submitted to IEEE:

> **"Temporal Dynamics as a Necessary Condition for Inverter Mode Detection in Datacenter Microgrids: An Ablation-Validated, Scenario-Split ML Study"**
> L. G. Engelmann — under review, IEEE Transactions on Smart Grid / IEEE Access

Key results:
- F1-macro = 0.988 (simulation, 5-seed mean, scenario-based split)
- Cross-scenario retention: 97.8% (train B+C → test A+D)
- Relay-optimized latency: 14.2 ms (200 trees, meets IEC 61850 GOOSE)
- External validation: NREL #253 (fuel cell inverter, 110 experiments)

---

## CI/CD

```
push → ruff lint → pytest → docker build (generator + dashboard)
```

Airflow DAGs run locally via `docker compose --profile local up`.
MLflow experiments tracked at `http://localhost:5000`.

---

## License

MIT — see [LICENSE](LICENSE)

---

*Built with Python 3.11, Streamlit, scikit-learn, XGBoost, SHAP, Apache Kafka, Apache Airflow, MLflow, PostgreSQL, MinIO, Grafana.*

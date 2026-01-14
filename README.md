# 🔧 Predictive Maintenance API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128.0-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg)](https://www.docker.com/)
[![Test Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen.svg)](https://github.com/foxymadeit/predictive-maintenance-dockerized-api)


## TL;DR

- Predictive maintenance system for industrial equipment failure detection.
- Gradient Boosting model (ROC-AUC 0.985, Recall 0.824)
- FastAPI + Docker + SHAP explainability
- 89% test coverage, CI/CD ready

---

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Problem & Solution](#-problem--solution)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Model Details](#-model-details)
- [Development](#-development)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Explainability](#-explainability)
- [License](#-license)

---

## Overview

This project implements an end-to-end machine learning system for **predictive maintenance** of industrial milling machines. The system predicts equipment failures before they occur, enabling proactive maintenance scheduling and reducing costly downtime.

**Business Value:**
- **Early failure detection** with 82% recall
- **Reliable alerts** with 72% precision  
- **Explainable predictions** using SHAP
- **Production-ready API** with Docker deployment
- **Real-time inference** with <100ms latency

---

## Key Features

### Machine Learning
- **Gradient Boosting Classifier** achieving 0.985 ROC-AUC
- **Custom threshold optimization** balancing precision and recall
- **SHAP-based explainability** for model transparency
- **Handles class imbalance** (failure rate ~3%)

### API Service
- **FastAPI** with automatic OpenAPI documentation
- **RESTful endpoints** for prediction and explanation
- **Batch prediction** support
- **SHAP visualization** endpoint
- **Health checks** and monitoring
- **Request ID tracking** for debugging

### Production Ready
- **Docker containerization** for consistent deployment
- **89% test coverage** (unit + integration tests)
- **Structured logging** (JSON format)
- **Configuration management** via YAML
- **CI/CD ready** with GitHub Actions workflow
- **Makefile** for common operations

---

## Problem & Solution

### Problem
Industrial equipment failures cause:
- Unplanned downtime
- Lost productivity
- Emergency repair costs
- Product quality issues

### Solution
**Predictive maintenance** system that:
1. Monitors equipment sensors in real-time
2. Predicts failures before they occur
3. Triggers maintenance alerts when risk is high
4. Provides explanations for each prediction

### Business Constraints
The model was selected based on **strict business requirements**:

**Hard Constraints:**
-  **Minimum Recall ≥ 0.80** — catch at least 80% of failures
-  **Minimum Precision ≥ 0.50** — keep false alarms manageable

**Ranking Metrics** (among models meeting constraints):
1. PR-AUC (primary)
2. ROC-AUC
3. Recall
4. Precision

---

## 📊 Model Performance

### Final Model: Gradient Boosting Classifier

**Operating Point:** Threshold = 0.10

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Recall** | 0.824 | Catches 82% of actual failures |
| **Precision** | 0.718 | 72% of alerts are true failures |
| **F1-Score** | 0.767 | Strong balance of precision/recall |
| **ROC-AUC** | 0.985 | Excellent ranking quality |
| **PR-AUC** | 0.839 | Best performance on imbalanced data |


### Why This Model?

**Only model** meeting both business constraints simultaneously  
**Highest PR-AUC** among eligible models  
**Best balance** between early detection and alert reliability  
**Production-ready** with calibrated threshold and explainability  

---

## 📁 Project Structure

```
predictive-maintenance-dockerized-api/
│
├── api/                          # FastAPI application
│   ├── main.py                   # API endpoints
│   ├── schemas.py                # Pydantic models
│   ├── deps.py                   # Dependency injection
│   └── static/                   # Frontend assets
│       └── index.html
│
├── src/                          # Core ML pipeline
│   ├── config.py                 # Configuration loader
│   ├── paths.py                  # Path management
│   ├── data_loader.py            # Data loading utilities
│   ├── preprocessing.py          # Feature engineering
│   ├── models.py                 # Model builders
│   ├── predictive_model.py       # Production model wrapper
│   ├── training.py               # Training pipeline
│   ├── evaluation.py             # Metrics calculation and evaluation
│   ├── thresholding.py           # Threshold optimization
│   ├── artifacts_io.py           # Model persistence
│   ├── logging_config.py         # Logging
│   └── visualization/            # Plotting utilities
│       ├── comparison.py
│       ├── explainability.py
│       └── threshold_analysis.py
│
├── tests/                        # Test suite (89% coverage)
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_eda.ipynb             # Exploratory analysis
│   ├── 02_baseline_models.ipynb # Baseline experiments
│   ├── 03_tree_models.ipynb     # Random Forest models
│   ├── 04_gradient_boosting_models.ipynb # Gradient Boosting models
│   └── 05_model_selection_and_explainability.ipynb # Final model selection
│
├── artifacts/                    # Model artifacts
│   ├── final/                    # Production model
│   │   ├── pipeline.joblib       # Trained pipeline
│   │   ├── threshold.joblib      # Decision threshold
│   │   ├── metrics.json          # Performance metrics
│   │   └── threshold_sweep.csv   # Threshold analysis
│   └── split/                    # Train/test split
│
├── data/
│   ├── raw/                      # Original dataset
│   └── processed/                # Processed data
│
├── config/
│   └── config.yml                # Project configuration
│
├── .github/workflows/
│   └── tests.yml                 # CI/CD pipeline
│
├── Dockerfile                    # Container definition
├── Makefile                      # Development commands
├── requirements.txt              # Python dependencies
├── pytest.ini                    # Test configuration
├── .coveragerc                   # Coverage settings
└── README.md                     # This file
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional, recommended)

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/foxymadeit/predictive-maintenance-dockerized-api.git
cd predictive-maintenance-dockerized-api

# Build and run
make build
make run-d

# Verify
make health
```

API runs at `http://localhost:8000`

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/foxymadeit/predictive-maintenance-dockerized-api.git
cd predictive-maintenance-dockerized-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Quick Test

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Air temperature [K]": 300,
    "Process temperature [K]": 310,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40,
    "Tool wear [min]": 100,
    "Type": "M"
  }'
```

**Response:**
```json
{
  "proba_failure": 0.156,
  "alert": 1,
  "threshold": 0.10
}
```

---

## API Documentation

### Interactive Docs

Once running, visit:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### Endpoints

#### `GET /`
Simple web interface for testing predictions.

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "threshold": 0.10,
  "features": ["Air temperature [K]", ...]
}
```

#### `POST /predict`
Single machine prediction.

**Request Body:**
```json
{
  "Air temperature [K]": 300.0,
  "Process temperature [K]": 310.0,
  "Rotational speed [rpm]": 1500.0,
  "Torque [Nm]": 40.0,
  "Tool wear [min]": 100.0,
  "Type": "M"
}
```

**Response:**
```json
{
  "proba_failure": 0.156,
  "alert": 1,
  "threshold": 0.10
}
```

**Fields:**
- `proba_failure`: Probability of failure (0-1)
- `alert`: Binary flag (0=safe, 1=maintenance needed)
- `threshold`: Decision threshold used

#### `POST /predict/batch`
Batch prediction for multiple machines.

**Request Body:**
```json
{
  "records": [
    {"Air temperature [K]": 300, ...},
    {"Air temperature [K]": 305, ...}
  ]
}
```

**Response:**
```json
{
  "results": [
    {"proba_failure": 0.156, "alert": 1, "threshold": 0.10},
    {"proba_failure": 0.043, "alert": 0, "threshold": 0.10}
  ]
}
```

#### `POST /explain`
Get SHAP explanation for a prediction.

**Query Parameters:**
- `top_k` (int, optional): Number of top features to return (default: 8)

**Request Body:** Same as `/predict`

**Response:**
```json
{
  "proba_failure": 0.156,
  "alert": 1,
  "threshold": 0.10,
  "top_contributors": [
    {
      "feature": "Torque [Nm]",
      "value": 40.0,
      "shap_value": 0.087,
      "direction": "increases_risk"
    },
    {
      "feature": "Tool wear [min]",
      "value": 100.0,
      "shap_value": 0.065,
      "direction": "increases_risk"
    },
    ...
  ]
}
```

#### `POST /explain/plot`
Get SHAP waterfall plot as PNG image.

**Request Body:** Same as `/predict`

**Response:** PNG image (Content-Type: image/png)

**Example:**
```bash
curl -X POST http://localhost:8000/explain/plot \
  -H "Content-Type: application/json" \
  -d '{"Air temperature [K]": 300, ...}' \
  -o shap_plot.png
```

---

## Model Details

### Features

**Numerical Features (5):**
- `Air temperature [K]` — ambient temperature
- `Process temperature [K]` — operational temperature  
- `Rotational speed [rpm]` — spindle rotation speed
- `Torque [Nm]` — torque measurement
- `Tool wear [min]` — cumulative tool usage time

**Categorical Features (1):**
- `Type` — machine quality variant (L=Low, M=Medium, H=High)

### Threshold Optimization

Decision threshold was selected by:
1. Computing precision-recall curve on test set
2. Filtering thresholds meeting business constraints:
   - Recall ≥ 0.80
   - Precision ≥ 0.50
3. Selecting threshold maximizing F1-score among valid options
4. Final threshold: **0.10** (optimized for early detection)

### Model Comparison

All trained models and their performance:

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|-----------|--------|----|---------| -------|
| **GB (final)** | **0.10** | **0.718** | **0.824** | **0.767** | **0.985** | **0.839** |
| GB (tuned) | 0.10 | 0.718 | 0.824 | 0.767 | 0.985 | 0.839 |
| RF (tuned) | 0.06 | 0.305 | 0.912 | 0.458 | 0.962 | 0.797 |
| LR (balanced) | 0.58 | 0.158 | 0.750 | 0.261 | 0.889 | 0.396 |
| LR (default) | 0.02 | 0.106 | 0.853 | 0.188 | 0.889 | 0.456 |

**Why Gradient Boosting?**
- Only model meeting both constraints
- Highest PR-AUC (critical for imbalanced data)
- Best precision-recall balance
- Stable performance across thresholds

---

## Development

### Setup Development Environment

```bash
# Clone and create venv
git clone https://github.com/foxymadeit/predictive-maintenance-dockerized-api.git
cd predictive-maintenance-dockerized-api
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest -v

# Run with coverage
pytest --cov=src --cov=api --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Makefile Commands

```bash
# Docker operations
make build              # Build Docker image
make run                # Run container (foreground)
make run-d              # Run container (background)
make stop               # Stop container
make rebuild            # Full rebuild cycle
make clean              # Remove images and cache

# API testing
make health             # Check API health
make predict            # Test /predict endpoint
make explain            # Test /explain endpoint
make explain-plot       # Test /explain/plot endpoint

# Testing
make test               # Run all tests
make test-cov           # Run tests with coverage
make test-unit          # Run unit tests only
make test-integration   # Run integration tests only
make test-docker        # Run tests in Docker
```

---

## Testing

### Test Coverage: 89%

```bash
# Run all tests
pytest -v

# With coverage report
pytest --cov=src --cov=api --cov-report=term-missing

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Test in Docker
make test-docker
```

### Test Structure

```
tests/
├── unit/                  # Unit tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   └── test_predictive_model.py
└── integration/           # Integration tests
    ├── test_api.py
    └── test_pipeline.py
```

## Explainability

### SHAP (SHapley Additive exPlanations)

The model uses **SHAP** to explain individual predictions:

**Global Explanations:**
- Feature importance ranking
- Average impact of each feature
- Feature interactions

**Local Explanations:**
- Contribution of each feature to a specific prediction
- Direction of impact (increases/decreases risk)
- Magnitude of effect

### Example Explanation

```bash
curl -X POST http://localhost:8000/explain?top_k=5 \
  -H "Content-Type: application/json" \
  -d '{
    "Air temperature [K]": 300,
    "Process temperature [K]": 310,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40,
    "Tool wear [min]": 100,
    "Type": "M"
  }'
```

**Response:**
```json
{
  "proba_failure": 0.156,
  "alert": 1,
  "threshold": 0.10,
  "top_contributors": [
    {
      "feature": "Torque [Nm]",
      "value": 40.0,
      "shap_value": 0.087,
      "direction": "increases_risk"
    },
    {
      "feature": "Tool wear [min]",
      "value": 100.0,
      "shap_value": 0.065,
      "direction": "increases_risk"
    },
    {
      "feature": "Rotational speed [rpm]",
      "value": 1500.0,
      "shap_value": -0.023,
      "direction": "decreases_risk"
    }
  ]
}
```

**Interpretation:**
> "The model predicts 15.6% failure probability (ALERT triggered). Primary risk factors: high torque (40 Nm) and elevated tool wear (100 min). Normal rotational speed slightly reduces risk."

### Visualization

Get SHAP waterfall plot:
```bash
curl -X POST http://localhost:8000/explain/plot \
  -H "Content-Type: application/json" \
  -d '{"Air temperature [K]": 300, ...}' \
  -o shap_waterfall.png
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Contact

**Project Author:** [@foxymadeit](https://github.com/foxymadeit)

**Project Link:** [https://github.com/foxymadeit/predictive-maintenance-dockerized-api](https://github.com/foxymadeit/predictive-maintenance-dockerized-api)
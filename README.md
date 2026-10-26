# ml-serving-demo

![CI](https://github.com/pujaux/ml-serving-demo/actions/workflows/ci.yml/badge.svg)

A small end-to-end ML project: train a scikit-learn model, serve it through a
FastAPI REST API, package it with Docker, and test and build it automatically
with GitHub Actions.

The model classifies iris flowers into three species (setosa, versicolor,
virginica) from four measurements. The dataset is simple on purpose. The goal
of the project is the **pipeline around the model**, not the model itself.

## How it works

```
train.py  ──►  models/iris.joblib  ──►  FastAPI app  ──►  POST /predict
 (train)         (saved model)          (loads model)      (JSON in, JSON out)
```

On every push, GitHub Actions retrains the model, runs the tests and builds the
Docker image.

## Tech stack

- **Python 3.12+**
- **scikit-learn**: Random Forest classifier
- **FastAPI + Uvicorn**: REST API with automatic docs
- **Pydantic**: request validation
- **pytest**: automated tests
- **Docker**: containerised deployment
- **GitHub Actions**: continuous integration

## Project structure

```
ml-serving-demo/
├── .github/workflows/ci.yml   # CI pipeline
├── app/
│   ├── __init__.py
│   └── main.py                # FastAPI app and /predict endpoint
├── tests/
│   └── test_api.py            # API tests
├── train.py                   # trains and saves the model
├── requirements.txt
├── Dockerfile
└── README.md
```

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (creates models/iris.joblib)
python train.py

# 4. Start the API
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the
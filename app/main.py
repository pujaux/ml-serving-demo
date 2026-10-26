from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "iris.joblib"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

app = FastAPI(title="Iris Classifier")
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Model not trained. Run: python train.py")
        _model = joblib.load(MODEL_PATH)
    return _model


class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: IrisFeatures):
    model = get_model()
    row = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]
    probs = model.predict_proba(row)[0]
    best = int(probs.argmax())
    return {
        "species": CLASS_NAMES[best],
        "probabilities": {n: round(float(p), 3) for n, p in zip(CLASS_NAMES, probs)},
    }
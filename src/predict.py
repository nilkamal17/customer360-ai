"""Load trained artifacts and score customer records."""
from pathlib import Path
import joblib
import pandas as pd

from src.features import MODEL_FEATURES, SEGMENT_FEATURES, validate_customer_frame

ROOT = Path(__file__).resolve().parents[1]
# Recall-oriented demo threshold; tune it using validation data and business costs.
CHURN_DECISION_THRESHOLD = 0.30
CHURN_MODEL_PATH = ROOT / "models" / "churn_model.joblib"
SEGMENT_MODEL_PATH = ROOT / "models" / "segment_model.joblib"


def load_models():
    missing = [str(path) for path in (CHURN_MODEL_PATH, SEGMENT_MODEL_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Model files are missing. Train the project first with 'python -m src.train'."
        )
    return joblib.load(CHURN_MODEL_PATH), joblib.load(SEGMENT_MODEL_PATH)


def score_customers(customers, churn_model=None, segment_model=None):
    """Return a copy with churn probability, risk band and cluster segment."""
    validate_customer_frame(customers)
    churn_model = churn_model or joblib.load(CHURN_MODEL_PATH)
    segment_model = segment_model or joblib.load(SEGMENT_MODEL_PATH)
    features = customers[MODEL_FEATURES].copy()
    output = customers.copy()
    probabilities = churn_model.predict_proba(features)[:, 1]
    output["churn_probability"] = probabilities.round(4)
    output["predicted_churn"] = (probabilities >= CHURN_DECISION_THRESHOLD).astype(int)
    output["risk_band"] = pd.cut(
        probabilities,
        bins=[-0.001, 0.30, 0.60, 1.001],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    output["segment"] = segment_model.predict(customers[SEGMENT_FEATURES]).astype(int)
    return output

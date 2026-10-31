"""Train churn classification and customer segmentation models."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    SEGMENT_FEATURES,
    TARGET_COLUMN,
    validate_customer_frame,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customers.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def build_churn_pipeline():
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessing), ("classifier", classifier)])


def build_segmentation_pipeline(n_clusters=4):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=42, n_init=20)),
        ]
    )


def train_models(data_path=DATA_PATH):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Run 'python scripts/generate_data.py' first."
        )

    frame = pd.read_csv(data_path)
    validate_customer_frame(frame, require_target=True)
    if frame[TARGET_COLUMN].nunique() < 2:
        raise ValueError("The churn column must contain both 0 and 1 labels.")

    X = frame[MODEL_FEATURES].copy()
    y = frame[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    churn_model = build_churn_pipeline()
    churn_model.fit(X_train, y_train)
    probabilities = churn_model.predict_proba(X_test)[:, 1]
    # A lower threshold is used in this demo to favour finding more at-risk
    # customers. A real business must choose this threshold using its costs.
    decision_threshold = 0.30
    predictions = (probabilities >= decision_threshold).astype(int)
    metrics = {
        "decision_threshold": decision_threshold,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "test_rows": int(len(y_test)),
        "dataset_rows": int(len(frame)),
        "synthetic_churn_rate": float(y.mean()),
        "note": "Metrics are from a synthetic demonstration dataset, not production customer data.",
    }

    segment_model = build_segmentation_pipeline(n_clusters=4)
    segment_model.fit(frame[SEGMENT_FEATURES])
    segment_labels = segment_model.predict(frame[SEGMENT_FEATURES])
    segmented = frame.copy()
    segmented["segment"] = segment_labels.astype(int)

    # Compute silhouette score only when the generated split has multiple clusters.
    scaled_segments = segment_model.named_steps["scaler"].transform(
        segment_model.named_steps["imputer"].transform(frame[SEGMENT_FEATURES])
    )
    sil_score = silhouette_score(scaled_segments, segment_labels)
    metrics["silhouette_score"] = float(sil_score)
    metrics["n_segments"] = int(len(np.unique(segment_labels)))

    profile_columns = [
        "tenure_months",
        "monthly_spend",
        "orders_last_90d",
        "avg_order_value",
        "days_since_last_purchase",
        "support_tickets_90d",
        "discount_usage_pct",
        "churned",
    ]
    profiles = segmented.groupby("segment")[profile_columns].mean().round(2).reset_index()
    profiles["customer_count"] = segmented.groupby("segment").size().values

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(churn_model, MODEL_DIR / "churn_model.joblib")
    joblib.dump(segment_model, MODEL_DIR / "segment_model.joblib")
    segmented.to_csv(DATA_PATH.parent / "customers_with_segments.csv", index=False)
    profiles.to_csv(REPORT_DIR / "segment_profiles.csv", index=False)
    with (REPORT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("Training completed successfully.")
    print(f"Churn model saved to: {MODEL_DIR / 'churn_model.joblib'}")
    print(f"Segmentation model saved to: {MODEL_DIR / 'segment_model.joblib'}")
    print("Evaluation metrics:")
    for name in ("accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "silhouette_score"):
        print(f"  {name}: {metrics[name]:.3f}")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Customer360 AI models")
    parser.add_argument("--data-path", default=str(DATA_PATH), help="Path to customers CSV")
    arguments = parser.parse_args()
    train_models(arguments.data_path)

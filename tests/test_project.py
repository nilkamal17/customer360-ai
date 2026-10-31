import pandas as pd
import pytest

from src.data_generator import generate_customers
from src.features import MODEL_FEATURES, SEGMENT_FEATURES, validate_customer_frame
from src.train import build_churn_pipeline, build_segmentation_pipeline


def test_generator_creates_expected_columns_and_binary_target():
    frame = generate_customers(n_customers=300, random_state=7)
    assert len(frame) == 300
    assert set(MODEL_FEATURES).issubset(frame.columns)
    assert set(frame["churned"].unique()).issubset({0, 1})
    assert frame["customer_id"].is_unique


def test_validate_customer_frame_reports_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_customer_frame(pd.DataFrame({"age": [30]}))


def test_churn_pipeline_can_fit_and_predict():
    frame = generate_customers(n_customers=400, random_state=8)
    model = build_churn_pipeline()
    model.fit(frame[MODEL_FEATURES], frame["churned"])
    predictions = model.predict(frame[MODEL_FEATURES].head(8))
    probabilities = model.predict_proba(frame[MODEL_FEATURES].head(8))
    assert len(predictions) == 8
    assert probabilities.shape == (8, 2)


def test_segmentation_pipeline_returns_four_clusters():
    frame = generate_customers(n_customers=300, random_state=9)
    model = build_segmentation_pipeline(n_clusters=4)
    labels = model.fit_predict(frame[SEGMENT_FEATURES])
    assert len(labels) == 300
    assert len(set(labels)) == 4

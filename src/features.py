"""Shared feature definitions and input validation for Customer360 AI."""

NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_spend",
    "orders_last_90d",
    "avg_order_value",
    "days_since_last_purchase",
    "support_tickets_90d",
    "complaints_90d",
    "discount_usage_pct",
    "payment_failures_90d",
]

CATEGORICAL_FEATURES = ["region", "membership_tier", "plan_type"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
SEGMENT_FEATURES = [
    "tenure_months",
    "monthly_spend",
    "orders_last_90d",
    "avg_order_value",
    "days_since_last_purchase",
    "support_tickets_90d",
    "discount_usage_pct",
]
TARGET_COLUMN = "churned"
ID_COLUMN = "customer_id"

CATEGORIES = {
    "region": ["North", "South", "East", "West", "Central"],
    "membership_tier": ["Basic", "Silver", "Gold", "Platinum"],
    "plan_type": ["Monthly", "Annual", "Pay-as-you-go"],
}


def validate_customer_frame(frame, require_target=False):
    """Raise a helpful ValueError when required customer columns are missing."""
    required = set(MODEL_FEATURES)
    if require_target:
        required.add(TARGET_COLUMN)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "The file is missing required columns: " + ", ".join(missing)
        )
    return frame

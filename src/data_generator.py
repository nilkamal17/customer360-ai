"""Create a reproducible synthetic customer dataset for learning and demos."""

from pathlib import Path
import numpy as np
import pandas as pd

from src.features import CATEGORIES


def generate_customers(n_customers=2500, random_state=42):
    """Return synthetic retail/subscription customer data with a churn label.

    This is simulated data. The churn label is generated from plausible risk
    signals for demonstration; it is not a measured real-world outcome.
    """
    if n_customers < 100:
        raise ValueError("n_customers must be at least 100 for a useful demo dataset.")

    rng = np.random.default_rng(random_state)
    age = np.clip(rng.normal(38, 12, n_customers).round(), 18, 80).astype(int)
    tenure = np.clip(rng.gamma(2.4, 10, n_customers).round(), 1, 72).astype(int)
    monthly_spend = np.clip(rng.lognormal(4.0, 0.65, n_customers), 8, 900).round(2)
    orders = np.clip(rng.poisson(5.0, n_customers), 0, 30).astype(int)
    avg_order = np.clip(monthly_spend / np.maximum(orders, 1) * rng.uniform(0.65, 1.35, n_customers), 5, 700).round(2)
    recency = np.clip(rng.gamma(2.1, 13, n_customers), 0, 180).round().astype(int)
    tickets = np.clip(rng.poisson(0.8, n_customers), 0, 10).astype(int)
    complaints = np.clip(rng.poisson(0.18, n_customers), 0, 5).astype(int)
    discount = np.clip(rng.beta(2.0, 5.0, n_customers) * 100, 0, 100).round(1)
    payment_failures = np.clip(rng.poisson(0.15, n_customers), 0, 5).astype(int)

    regions = rng.choice(CATEGORIES["region"], n_customers, p=[0.22, 0.23, 0.18, 0.20, 0.17])
    tiers = rng.choice(CATEGORIES["membership_tier"], n_customers, p=[0.46, 0.29, 0.18, 0.07])
    plans = rng.choice(CATEGORIES["plan_type"], n_customers, p=[0.48, 0.32, 0.20])

    # Synthetic probability: higher recency, payment issues, complaints and
    # support burden increase the chance of a churn label.
    logit = (
        -3.10
        + 0.038 * recency
        - 0.14 * orders
        - 0.045 * tenure
        + 0.72 * tickets
        + 1.25 * complaints
        + 1.05 * payment_failures
        + 0.014 * discount
        + np.where(plans == "Monthly", 0.72, 0.0)
        + np.where(plans == "Annual", -0.55, 0.0)
        + np.where(tiers == "Platinum", -0.65, 0.0)
        + rng.normal(0, 0.18, n_customers)
    )
    probabilities = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, probabilities)

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(1, n_customers + 1)],
            "age": age,
            "tenure_months": tenure,
            "monthly_spend": monthly_spend,
            "orders_last_90d": orders,
            "avg_order_value": avg_order,
            "days_since_last_purchase": recency,
            "support_tickets_90d": tickets,
            "complaints_90d": complaints,
            "discount_usage_pct": discount,
            "payment_failures_90d": payment_failures,
            "region": regions,
            "membership_tier": tiers,
            "plan_type": plans,
            "churned": churned,
        }
    )


def save_demo_dataset(output_path=None, n_customers=2500, random_state=42):
    root = Path(__file__).resolve().parents[1]
    destination = Path(output_path) if output_path else root / "data" / "customers.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_customers(n_customers=n_customers, random_state=random_state)
    frame.to_csv(destination, index=False)
    return destination, frame

"""Streamlit dashboard for Customer360 AI."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

from src.features import CATEGORIES, MODEL_FEATURES, SEGMENT_FEATURES, validate_customer_frame
from src.predict import load_models, score_customers

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "customers.csv"
PROFILE_PATH = ROOT / "reports" / "segment_profiles.csv"
METRICS_PATH = ROOT / "reports" / "metrics.json"

st.set_page_config(page_title="Customer360 AI", page_icon="👥", layout="wide")
st.title("👥 Customer360 AI")
st.caption("Customer segmentation and churn prediction | Machine learning portfolio project")


def show_setup_message():
    st.warning("Project models are not ready yet. Run these commands in the VS Code terminal:")
    st.code("python scripts/generate_data.py\npython -m src.train\nstreamlit run app.py", language="bash")


def load_base_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


try:
    churn_model, segment_model = load_models()
except FileNotFoundError:
    show_setup_message()
    st.stop()

base_data = load_base_data()
if base_data is None:
    st.error("data/customers.csv was not found. Run python scripts/generate_data.py first.")
    st.stop()

metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Single Customer Check", "Batch Customer Scoring", "Customer Segments", "Model Performance"],
)

if page == "Overview":
    st.subheader("Know your customers. Retain the right customers.")
    st.write(
        "Customer360 AI combines an unsupervised K-Means clustering model to group similar "
        "customers with a supervised Random Forest model to estimate churn risk."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers in demo data", f"{len(base_data):,}")
    col2.metric("Customer segments", str(metrics.get("n_segments", 4)))
    col3.metric("Test ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    col4.metric("Synthetic churn rate", f"{base_data['churned'].mean():.1%}")
    st.info("This starter project uses simulated customer data. Predictions are for learning and demonstration, not real retention decisions.")
    st.subheader("Sample customer records")
    st.dataframe(base_data.head(12), use_container_width=True)

elif page == "Single Customer Check":
    st.subheader("Estimate churn risk for one customer")
    with st.form("single_customer_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", min_value=18, max_value=100, value=35)
        tenure = c1.number_input("Tenure (months)", min_value=1, max_value=120, value=18)
        monthly_spend = c1.number_input("Monthly spend", min_value=0.0, max_value=10000.0, value=120.0)
        orders = c2.number_input("Orders in last 90 days", min_value=0, max_value=100, value=5)
        avg_order = c2.number_input("Average order value", min_value=0.0, max_value=10000.0, value=45.0)
        recency = c2.number_input("Days since last purchase", min_value=0, max_value=365, value=20)
        tickets = c3.number_input("Support tickets (90 days)", min_value=0, max_value=30, value=1)
        complaints = c3.number_input("Complaints (90 days)", min_value=0, max_value=20, value=0)
        discount = c3.number_input("Discount usage (%)", min_value=0.0, max_value=100.0, value=20.0)
        failures = c3.number_input("Payment failures (90 days)", min_value=0, max_value=20, value=0)
        region = c1.selectbox("Region", CATEGORIES["region"])
        tier = c2.selectbox("Membership tier", CATEGORIES["membership_tier"])
        plan = c3.selectbox("Plan type", CATEGORIES["plan_type"])
        submitted = st.form_submit_button("Predict customer risk")

    if submitted:
        customer = pd.DataFrame([{
            "age": age,
            "tenure_months": tenure,
            "monthly_spend": monthly_spend,
            "orders_last_90d": orders,
            "avg_order_value": avg_order,
            "days_since_last_purchase": recency,
            "support_tickets_90d": tickets,
            "complaints_90d": complaints,
            "discount_usage_pct": discount,
            "payment_failures_90d": failures,
            "region": region,
            "membership_tier": tier,
            "plan_type": plan,
        }])
        result = score_customers(customer, churn_model, segment_model).iloc[0]
        left, middle, right = st.columns(3)
        left.metric("Churn probability", f"{result['churn_probability']:.1%}")
        middle.metric("Risk band", str(result["risk_band"]))
        right.metric("Customer segment", f"Segment {result['segment']}")
        if result["risk_band"] == "High":
            st.error("High estimated churn risk. Review this customer before taking action.")
        elif result["risk_band"] == "Medium":
            st.warning("Medium estimated churn risk. Consider monitoring engagement.")
        else:
            st.success("Low estimated churn risk under this demonstration model.")
        st.caption("Risk bands are illustrative thresholds, not validated business policy.")

elif page == "Batch Customer Scoring":
    st.subheader("Score a CSV of customers")
    st.write("Upload a CSV containing all required model input columns. The `churned` target column is not required.")
    st.download_button(
        "Download sample customer CSV",
        base_data[MODEL_FEATURES].head(100).to_csv(index=False).encode("utf-8"),
        file_name="customer360_sample_input.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is not None:
        try:
            customer_frame = pd.read_csv(uploaded)
            validate_customer_frame(customer_frame)
            scored = score_customers(customer_frame, churn_model, segment_model)
            st.success(f"Scored {len(scored):,} customers.")
            st.dataframe(scored, use_container_width=True)
            st.download_button(
                "Download scored customers",
                scored.to_csv(index=False).encode("utf-8"),
                file_name="customer360_scored_customers.csv",
                mime="text/csv",
            )
        except Exception as error:
            st.error(f"Could not score this file: {error}")

elif page == "Customer Segments":
    st.subheader("Explore customer groups")
    segmented_path = ROOT / "data" / "customers_with_segments.csv"
    if not segmented_path.exists():
        st.warning("Run the training command to generate segment labels.")
    else:
        segmented = pd.read_csv(segmented_path)
        if PROFILE_PATH.exists():
            profiles = pd.read_csv(PROFILE_PATH)
            st.write("Average customer profile by K-Means segment")
            st.dataframe(profiles, use_container_width=True)
            chart_data = profiles.set_index("segment")[["monthly_spend", "orders_last_90d", "days_since_last_purchase"]]
            st.bar_chart(chart_data)
        st.write("Customers by segment")
        counts = segmented["segment"].value_counts().sort_index().rename("customers")
        st.bar_chart(counts)
        st.dataframe(segmented.head(20), use_container_width=True)

elif page == "Model Performance":
    st.subheader("Evaluation on a held-out test set")
    st.caption("Results are from the generated synthetic dataset and should not be interpreted as real-world performance.")
    metric_cols = st.columns(5)
    for column, key, label in zip(metric_cols, ["precision", "recall", "f1_score", "roc_auc", "pr_auc"], ["Precision", "Recall", "F1 score", "ROC-AUC", "PR-AUC"]):
        column.metric(label, f"{metrics.get(key, 0):.3f}")
    cm = metrics.get("confusion_matrix")
    if cm:
        st.write("Confusion matrix (rows = actual, columns = predicted; class 0 = stayed, class 1 = churned)")
        st.dataframe(pd.DataFrame(cm, index=["Actual: stayed", "Actual: churned"], columns=["Predicted: stayed", "Predicted: churned"]), use_container_width=True)
    st.metric("K-Means silhouette score", f"{metrics.get('silhouette_score', 0):.3f}")
    st.json({key: value for key, value in metrics.items() if key != "classification_report"})

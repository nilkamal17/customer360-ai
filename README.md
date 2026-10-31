# Customer360 AI — Customer Segmentation & Churn Prediction

A beginner-friendly, end-to-end machine learning portfolio project that combines:

1. **Customer segmentation (unsupervised learning):** K-Means groups customers with similar engagement and spending patterns.
2. **Churn prediction (supervised learning):** a Random Forest classifier estimates the probability that a customer may leave.
3. **Interactive dashboard:** Streamlit lets a user inspect customer groups, score one customer, or upload a CSV for batch scoring.

> **Important data note:** The included `data/customers.csv` is synthetic demo data generated for this project. The churn label is simulated from illustrative customer signals. It is not real bank, retailer, or customer data, and model metrics are not evidence of real-world performance. Do not use these predictions to make real customer decisions without real, representative data, validation, privacy review, and human oversight.

---

## 1. Business problem

A subscription or retail business may have thousands of customers, but they do not all behave in the same way. Teams need to understand customer groups and identify customers who may be at risk of leaving.

Customer360 AI demonstrates a simple workflow:

- Group customers by tenure, spending, purchase frequency, recency, and engagement.
- Estimate each customer's churn probability.
- Prioritise high-risk customers for human review and appropriate retention experiments.

It is a portfolio prototype, not a production customer relationship management system.

## 2. Technology stack

| Area | Technology |
|---|---|
| Language | Python 3.10–3.12 (3.11 recommended) |
| Data handling | Pandas, NumPy |
| Churn model | Scikit-learn Random Forest classifier |
| Segmentation | Scikit-learn K-Means clustering |
| Preprocessing | SimpleImputer, StandardScaler, OneHotEncoder |
| Model storage | Joblib |
| Dashboard | Streamlit |
| Tests | Pytest |
| Version control | Git and GitHub |

## 3. Project structure

```text
Customer360_AI/
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Python dependencies
├── README.md                      # Tutorial and documentation
├── .gitignore                     # Files not to upload to GitHub
├── pytest.ini                     # Pytest project configuration
├── data/
│   ├── customers.csv              # Included synthetic demo dataset
│   ├── customers_with_segments.csv # Created during training
│   └── README.md                  # Dataset explanation
├── models/
│   ├── churn_model.joblib         # Saved churn pipeline
│   └── segment_model.joblib       # Saved K-Means pipeline
├── reports/
│   ├── metrics.json               # Model evaluation metrics
│   └── segment_profiles.csv       # Average profile by cluster
├── scripts/
│   └── generate_data.py           # Generate or regenerate data
├── src/
│   ├── __init__.py
│   ├── data_generator.py          # Synthetic data generation
│   ├── features.py                # Shared feature definitions/validation
│   ├── train.py                   # Train and evaluate both models
│   └── predict.py                 # Load models and score customers
└── tests/
    └── test_project.py            # Automated tests
```

## 4. VS Code setup — Windows CMD

### Step 1: Extract and open the project

1. Download and extract the `Customer360_AI.zip` file.
2. Open VS Code.
3. Select **File → Open Folder**.
4. Choose the extracted `Customer360_AI` folder. It must be the folder containing `app.py` and `requirements.txt`.
5. Open a new terminal. The commands below use **Command Prompt (CMD)**.

### Step 2: Check Python

```cmd
python --version
```

Python 3.11 is recommended. If `python` is not recognised, try:

```cmd
py --version
```

### Step 3: Create a virtual environment

```cmd
python -m venv venv
```

If your computer uses the Python launcher instead:

```cmd
py -3.11 -m venv venv
```

### Step 4: Activate it

```cmd
venv\Scripts\activate
```

Your terminal should now begin with `(venv)`.

### Step 5: Upgrade pip and install libraries

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**If you see `requirements.txt` not found:** run `cd` to view the current folder and `dir` to list files. Navigate to the extracted project folder (the one that contains `requirements.txt`) before running the install command. Keep the environment activated while you do this.

## 5. Generate data and train the models

A synthetic example dataset is already included, so you can start without downloading any external dataset.

### Option A — use the included dataset

From the project root, run:

```cmd
python -m src.train
```

### Option B — regenerate the dataset first

```cmd
python scripts\generate_data.py
python -m src.train
```

To create more rows:

```cmd
python scripts\generate_data.py --rows 5000
python -m src.train
```

Training creates or refreshes:

- `models/churn_model.joblib`
- `models/segment_model.joblib`
- `data/customers_with_segments.csv`
- `reports/metrics.json`
- `reports/segment_profiles.csv`

The churn model uses a 0.30 demo probability threshold to favour recall (finding more potentially at-risk customers). This is an illustrative operating point, not a universal business rule. A real organisation should select a threshold using validation data, the cost of false positives and false negatives, and its retention process.

## 6. Launch the dashboard

```cmd
streamlit run app.py
```

Streamlit should open a browser tab. Otherwise, visit:

```text
http://localhost:8501
```

Dashboard sections:

- **Overview:** dataset size, churn rate, and model snapshot.
- **Single Customer Check:** enter customer attributes and estimate churn risk and segment.
- **Batch Customer Scoring:** upload a CSV and download scored results.
- **Customer Segments:** inspect K-Means group profiles and counts.
- **Model Performance:** inspect precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, and silhouette score.

For batch upload, the CSV must include the following columns (the `churned` target column is not needed):

```text
age, tenure_months, monthly_spend, orders_last_90d, avg_order_value,
days_since_last_purchase, support_tickets_90d, complaints_90d,
discount_usage_pct, payment_failures_90d, region, membership_tier, plan_type
```

Use the dashboard's **Download sample customer CSV** button to get a correctly formatted example.

## 7. Run automated tests

```cmd
pytest -q
```

The tests check data generation, required-column validation, churn pipeline predictions, and the four-cluster segmentation pipeline.

## 8. Upload to GitHub

Create a new, empty GitHub repository named `customer360-ai` (do not initialise it with a README, `.gitignore`, or licence because this project already contains its own files).

In the VS Code CMD terminal, confirm you are at the project root, then run:

```cmd
git init
git add .
git status
git commit -m "Initial commit - Customer360 AI"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/customer360-ai.git
git push -u origin main
```

Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username. If the remote already exists, use this instead of `git remote add origin`:

```cmd
git remote set-url origin https://github.com/YOUR_GITHUB_USERNAME/customer360-ai.git
```

For future updates:

```cmd
git add .
git commit -m "Describe your update"
git push
```

Do not upload your `venv/` folder. It is excluded in `.gitignore`.

## 9. How the machine learning works

### Churn prediction — Random Forest

- The target is `churned` (0 = stayed, 1 = churned) in the simulated training data.
- Numeric values are imputed and scaled.
- Categorical values are imputed and one-hot encoded.
- A Random Forest classifier learns patterns from a training split.
- The held-out test split is used to calculate classification metrics.

### Customer segmentation — K-Means

- K-Means uses selected customer behaviour features such as tenure, spend, order frequency, recency, and support activity.
- Inputs are imputed and standardised before clustering.
- The demo uses four clusters. Cluster numbers are arbitrary IDs; Segment 0 is not inherently better or worse than Segment 1.
- Segment profiles are exported so a business analyst can interpret the groups.

### Metrics to understand

- **Precision:** Of customers flagged as likely to churn, how many were labelled churned in the test data?
- **Recall:** Of customers labelled churned in the test data, how many did the model flag?
- **F1:** A combined measure of precision and recall.
- **ROC-AUC / PR-AUC:** How well the model ranks positive cases across thresholds. PR-AUC is useful to inspect when churn cases are less common.
- **Silhouette score:** A clustering diagnostic; closer to 1 generally indicates better-separated clusters, while values near 0 indicate overlapping groups. It is not a business-success measure.

## 10. Improvements for an advanced portfolio version

1. Compare Logistic Regression, Random Forest, and Gradient Boosting on the same time-aware test split.
2. Add SHAP or permutation-based explanations for individual churn scores.
3. Build a retention-action experiment and measure incremental lift, rather than assuming a prediction itself reduces churn.
4. Add monitoring for data drift and changes in churn prevalence.
5. Validate on real, permissioned, representative data and document privacy, fairness, and model limitations.

## Troubleshooting

- **`No module named src`:** Open the actual project root in VS Code and run commands from that folder. For training, use `python -m src.train` exactly.
- **`requirements.txt` not found:** run `cd`, then `dir`; navigate to the folder containing `requirements.txt`.
- **`streamlit` not recognised:** activate `venv`, then run `python -m pip install -r requirements.txt` again. You can also launch with `python -m streamlit run app.py`.
- **Models missing:** run `python -m src.train` before starting the dashboard.
- **Git says remote already exists:** use `git remote set-url origin ...` rather than adding `origin` again.

---

Made as an educational AI/ML portfolio project. Always explain that the included dataset and labels are synthetic when presenting this project.
"# customer360-ai" 

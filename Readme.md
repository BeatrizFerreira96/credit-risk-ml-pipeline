# Credit Risk Prediction Pipeline with SQL Backend, XGBoost, and Query Optimization

# Credit Risk Prediction Pipeline with SQL Backend, XGBoost, and Query Optimization

## Overview

This project builds an end-to-end machine learning pipeline for predicting customer credit risk using a structured SQL backend and gradient boosting models.

The workflow covers the full lifecycle of a practical ML system:

* Raw tabular data ingestion into SQLite
* SQL querying and dataset retrieval
* Data preprocessing with scikit-learn Pipelines
* Model training with XGBoost
* Hyperparameter tuning
* Feature importance analysis
* SQL query optimization with indexes

This project emphasizes **real-world ML engineering**, not just model training.

---

## Business Problem

Financial institutions need to estimate the probability that a borrower may default on credit obligations.

Accurate credit risk models help with:

* Loan approval decisions
* Risk-based pricing
* Portfolio monitoring
* Regulatory compliance
* Loss reduction

---

## Dataset

Used the German Credit Dataset, a widely used benchmark dataset for credit risk classification.

### Dataset Characteristics

* 1000 customer records
* 20 input features
* Binary target variable:

  * 0 = good credit risk
  * 1 = bad credit risk

Features include:

* Account status
* Credit duration
* Credit history
* Savings
* Employment status
* Installment rate
* Personal attributes
* Existing credits
* Housing information

---

## Tech Stack

* Python
* SQLite
* pandas
* scikit-learn
* XGBoost
* VS Code
* WSL Ubuntu

---

## Project Structure

```text
credit-risk-ml-pipeline/
├── data/
│   ├── raw/
│   │   └── credit_data.csv
│   └── credit.db
│
├── database/
│   └── schema.sql
│
├── src/
│   ├── data_loader.py
│   ├── queries.py
│   └── train.py
│
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Workflow

### 1. Data Ingestion

Raw dataset loaded from CSV into SQLite database.

```python
df.to_sql("credit_data", conn, if_exists="replace", index=False)
```

### 2. SQL Backend

Data stored in relational format and queried using SQL.

Example:

```sql
SELECT *
FROM credit_data
WHERE target = 1;
```

### 3. Data Preprocessing

Used scikit-learn Pipeline and ColumnTransformer for:

* One-hot encoding categorical variables
* Passing numeric columns directly
* Reproducible preprocessing

### 4. Model Training

Used XGBoost classifier for binary classification.

### 5. Model Evaluation

Measured performance using:

* Accuracy
* ROC-AUC

### 6. Explainability

Extracted feature importances to identify main credit risk drivers.

### 7. Query Optimization

Created indexes in SQLite and validated optimizer behavior.

---

## Machine Learning Results

### Baseline Model

| Metric   | Score |
| -------- | ----- |
| Accuracy | 0.770 |
| ROC-AUC  | 0.795 |

### Tuned Model

| Metric   | Score |
| -------- | ----- |
| Accuracy | 0.745 |
| ROC-AUC  | 0.781 |

### Interpretation

Hyperparameter tuning did not outperform the baseline model, demonstrating the importance of validation rather than assuming tuning always improves performance.

---

## Top Predictive Features

The model identified several high-impact variables related to default risk, including:

* Checking account status
* Credit duration
* Savings level
* Prior credit history
* Financial commitments

This aligns with expected drivers of borrower risk.

---

## SQL Performance Optimization

Created indexes on frequently queried columns:

```sql
CREATE INDEX idx_target ON credit_data(target);
CREATE INDEX idx_duration ON credit_data(feature_1);
CREATE INDEX idx_status ON credit_data(feature_0);
```

Validated query planner usage:

```sql
EXPLAIN QUERY PLAN
SELECT * FROM credit_data WHERE target = 1;
```

Output:

```text
SEARCH TABLE credit_data USING INDEX idx_target
```

This confirms indexed query execution instead of full table scans.

---

## Key Skills Demonstrated

### Machine Learning

* Classification modeling
* Gradient boosting
* Feature importance
* Model evaluation
* Hyperparameter tuning

### Data Engineering

* SQL database design
* Data ingestion pipelines
* Query optimization
* Structured data retrieval

### Software Engineering

* Modular code organization
* Reproducible pipelines
* Clean project structure

---

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Load Data

```bash
python3 src/data_loader.py
```

### Train Model

```bash
python3 -m src.train
```

---

## Future Improvements

* Cross-validation with repeated folds
* SHAP explainability analysis
* Class imbalance strategies
* Model deployment with FastAPI
* Dockerized environment
* Larger production-scale datasets

---

## Why This Project Matters

Many ML projects focus only on training a model.

This project demonstrates the broader system required in real applications:

**database + preprocessing + modeling + optimization + reproducibility**

That reflects how machine learning is used in production environments.

---

## Author

Developed as a portfolio project focused on practical machine learning systems, credit risk modeling, and SQL-backed pipelines.

# Credit Risk Prediction Web Application with FastAPI, XGBoost, and SQL Backend

## Overview

This project is an end-to-end machine learning web application for predicting customer credit risk using an XGBoost classifier trained on the German Credit Dataset.

The application includes a deployed FastAPI backend and an interactive frontend interface for real-time credit risk prediction.

The workflow covers the full lifecycle of a practical ML system:

- Raw tabular data ingestion into SQLite
- SQL querying and dataset retrieval
- Data preprocessing with scikit-learn Pipelines
- Model training with XGBoost
- Hyperparameter tuning
- Feature importance analysis
- SQL query optimization with indexes
- REST API deployment with FastAPI
- Interactive frontend integration

This project emphasizes real-world ML engineering, not just model training.

---

## Live Demo

https://credit-risk-ml-pipeline.onrender.com

---

## Business Problem

Financial institutions need to estimate the probability that a borrower may default on credit obligations.

Accurate credit risk models help with:

- Loan approval decisions
- Risk-based pricing
- Portfolio monitoring
- Regulatory compliance
- Loss reduction

---

## Dataset

This project uses the German Credit Dataset, a widely used benchmark dataset for credit risk classification.

### Dataset Characteristics

- 1000 customer records
- 20 input features
- Binary target variable:
  - 0 = good credit risk
  - 1 = bad credit risk

Features include:

- Account status
- Credit duration
- Credit history
- Savings
- Employment status
- Installment rate
- Personal attributes
- Existing credits
- Housing information

---

## Tech Stack

### Backend
- FastAPI
- Python
- scikit-learn
- XGBoost

### Frontend
- HTML
- CSS
- JavaScript

### Database
- SQLite

### Deployment
- Render

### Development Environment
- VS Code
- WSL Ubuntu

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
│   ├── train.py
│   └── model.pkl
│
├── main.py
├── index.html
├── requirements.txt
└── README.md
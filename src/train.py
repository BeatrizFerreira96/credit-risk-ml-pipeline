import pandas as pd
from src.queries import get_credit_data

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

from xgboost import XGBClassifier

# Load data
df = get_credit_data()

# Features / target
X = df.drop(columns=["target"])
y = df["target"]

# Split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Detect column types
categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# Pipeline
pipeline = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", XGBClassifier(
        eval_metric="logloss",
        random_state=42
    ))
])

# Hyperparameter search
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 4, 5, 6],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.8, 0.9, 1.0]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=10,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Fit AFTER split
search.fit(X_train, y_train)

best_model = search.best_estimator_

# Evaluate
pred = best_model.predict(X_test)
proba = best_model.predict_proba(X_test)[:, 1]

print("Best Params:", search.best_params_)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("ROC-AUC :", round(roc_auc_score(y_test, proba), 4))


model = best_model.named_steps["model"]
feature_names = best_model.named_steps["prep"].get_feature_names_out()

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print(importance_df.head(15))
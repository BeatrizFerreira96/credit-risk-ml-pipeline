import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

column_names = [
    "checking_status", "duration", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment",
    "installment_commitment", "personal_status", "other_parties",
    "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits",
    "job", "num_dependents", "own_telephone", "foreign_worker",
    "target"
]
#load data
df = pd.read_csv("../data/raw/credit_data.csv", sep="\s+", header=None)
df.columns = column_names

# Features / target
X = df.drop(columns=["target"])
y = df["target"]
# Convert labels to 0/1 (clean!)
y = y.replace({1: 0, 2: 1})



# Detect categorical columns
categorical_cols = X.select_dtypes(include="object").columns
numeric_cols = X.select_dtypes(exclude="object").columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# Full pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
))
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
pipeline.fit(X_train, y_train)

# Save entire pipeline
joblib.dump(pipeline, "model.pkl")

print("Pipeline trained and saved!")
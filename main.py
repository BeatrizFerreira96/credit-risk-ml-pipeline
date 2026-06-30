from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import shap

app = FastAPI()   # ✅ MUST come before decorators
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
model = joblib.load("src/model.pkl")
print(model.named_steps)


# ✅ Input schema
class InputData(BaseModel):
    checking_status: str
    duration: int
    credit_history: str
    purpose: str
    credit_amount: int
    savings_status: str
    employment: str
    installment_commitment: int
    age: int
    existing_credits: int

    # optional fields your model expects
    personal_status: Optional[str] = "A93"  # default value

@app.get("/")
def root():
    import os
    print("Serving file from:", os.getcwd())
    return FileResponse("templates/index.html")


@app.post("/predict")
def predict(customer: InputData):

    try:
        data = pd.DataFrame([customer.dict()])

        categorical_defaults = {
            "personal_status": "A93",
            "other_parties": "A101",
            "property_magnitude": "A121",
            "other_payment_plans": "A141",
            "housing": "A151",
            "job": "A171",
            "own_telephone": "A191",
            "foreign_worker": "A201"
        }

        numeric_defaults = {
            "residence_since": 2,
            "num_dependents": 1
        }

        for col, value in categorical_defaults.items():
            if col not in data:
                data[col] = value

        for col, value in numeric_defaults.items():
            if col not in data:
                data[col] = value

        # preprocess input
        processed_data = model.named_steps["preprocessor"].transform(data)

        pred = model.named_steps["classifier"].predict(processed_data)[0]

        prob = model.named_steps["classifier"].predict_proba(processed_data)[0][1]
        explainer = shap.TreeExplainer(model.named_steps["classifier"])
        shap_values = explainer(processed_data)

        feature_importance = {}

        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        for name, value in zip(feature_names, shap_values.values[0]):
            feature_importance[name] = round(float(value), 3)
        
        proba = model.named_steps["classifier"].predict_proba(processed_data)[0]

        pred = model.named_steps["classifier"].predict(processed_data)[0]

        prob = proba[1]

        confidence = max(proba)
        
        return {
    "prediction": "high risk" if pred == 1 else "low risk",
    "risk_probability": round(float(prob), 3),
    "confidence": round(float(confidence), 3),
    "shap_values": feature_importance
    }

    except Exception as e:
        return {"error": str(e)}